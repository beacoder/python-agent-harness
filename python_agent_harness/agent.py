"""Core agent loop: FSM-driven send -> tool -> supervise.

Implements the gptel-agent-harness supervision semantics:

- context ratio is computed before each top-level WAIT; compaction is
  triggered past the threshold (top-level, agentic, not already compacting)
- queued plan/build-mode prompts are injected before sending
- terminal states on an agentic top-level loop nudge the model back to
  WAIT while nudge budget remains (max 2), reset on tool calls
- tool results are sanitized (None -> error placeholder, non-str -> str)
- tool-call batches never strand the loop: failures become error results
- token calibration is updated from API-reported input tokens
- sessions are auto-saved after each response
"""

from __future__ import annotations

import json
from typing import Any

from . import config
from .compaction import insert_compact_frame, last_user_request, read_prompt_file
from .fsm import Fsm, FsmInfo, PendingToolCall, State, Supervisor, sanitize_tool_result
from .models import Message, ToolCall
from .tokenizer import context_window_for, estimate_payload_tokens


class AgentLoop:
    """Runs one agent session until terminal or max rounds."""

    def __init__(
        self,
        session: Any,
        messages: list[Message] | None = None,
        top_level: bool = True,
        system: str | None = None,
        max_rounds: int = 60,
    ) -> None:
        self.session = session
        self.messages: list[Message] = messages if messages is not None else []
        self.top_level = top_level
        self.system = system
        self.max_rounds = max_rounds
        self.info = FsmInfo(
            buffer=session,
            model=session.model,
            backend=session.backend,
            tools=bool(session.tools_enabled),
            top_level=top_level,
        )
        self.fsm = Fsm(info=self.info)
        self.supervisor = Supervisor(session)

    # ------------------------------------------------------------------
    # context management
    # ------------------------------------------------------------------
    def _update_context_ratio(self) -> None:
        raw = estimate_payload_tokens(
            self.system, [m.to_api() for m in self.messages],
            [t.to_api() for t in self.session.tool_specs()],
        )
        self.session.calibrator.last_raw_estimate = raw
        calibrated = self.session.calibrator.calibrate(raw)
        window = context_window_for(self.session.model)
        self.session.context_ratio = calibrated / float(window)
        self.session.notify("context")

    def _need_compaction(self) -> bool:
        return (
            self.top_level
            and self.session.tools_enabled
            and not self.session.compacting
            and self.session.context_ratio is not None
            and self.session.context_ratio > config.CONTEXT_TRIGGER
        )

    # ------------------------------------------------------------------
    # prompt injection (plan/build mode)
    # ------------------------------------------------------------------
    def _inject_pending_prompts(self) -> None:
        if not self.top_level:
            if self.session.plan_mode.is_plan and not self.info.harness_injected:
                self.messages.insert(
                    len(self.messages),
                    Message(role="user", content=self.session.plan_mode.plan_reminder()),
                )
                self.info.harness_injected = True
            return
        prompts = self.session.plan_mode.consume_prompts()
        prompts = prompts + list(self.session.pending_user_prompts)
        self.session.pending_user_prompts = []
        if not prompts:
            return
        # inject before the last user message when it is a plain request;
        # otherwise append (tool result last -> must not split call/result)
        insert_at = len(self.messages)
        if self.messages:
            last = self.messages[-1]
            if last.role == "user" and isinstance(last.content, str) and not last.tool_call_id:
                insert_at = len(self.messages) - 1
        for i, text in enumerate(prompts):
            self.messages.insert(insert_at + i, Message(role="user", content=text))

    # ------------------------------------------------------------------
    # compaction
    # ------------------------------------------------------------------
    def compact(self) -> bool:
        """Compact the conversation; return True on success."""
        request = last_user_request([m.to_api() for m in self.messages])
        if not request:
            return False
        self.session.compacting = True
        try:
            conversation = "\n\n".join(
                f"{m.role}: {m.text()}" for m in self.messages if m.text()
            )
            system = read_prompt_file("compact.txt")
            resp, _ = self.session.client.chat_sync(
                [Message(role="user", content=conversation)], system=system
            )
            summary = resp.text()
            if not summary:
                return False
            self.session.cache.reset_epoch()
            frame = config.COMPACT_HEADER + summary + config.COMPACT_SEPARATOR
            self.messages = [
                Message(role="system", content=frame.strip()),
                Message(role="user", content=request),
            ]
            self.session.notify("compact")
            return True
        except Exception as e:  # noqa: BLE001 - compaction failure is non-fatal
            self.session.notify("error")
            self.session.log(f"compaction failed: {e}")
            return False
        finally:
            self.session.compacting = False

    # ------------------------------------------------------------------
    # tool execution
    # ------------------------------------------------------------------
    def _execute_tool_call(self, call: ToolCall) -> str:
        args = call.arguments
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if not isinstance(args, dict):
            args = {}
        return self.session.execute_tool(call.name, args, call_id=call.id)

    def _run_tool_round(self) -> None:
        """Execute all pending tool calls; deliver results as messages."""
        pending = list(self.info.pending)
        if not pending:
            return
        self.messages.append(
            Message(role="assistant", content="", tool_calls=[p.call for p in pending])
        )
        for p in pending:
            result = sanitize_tool_result(self._execute_tool_call(p.call))
            p.call.result = result
            if hasattr(self.session, "take_diff"):
                p.call.diff = self.session.take_diff(p.call.id)
            self.messages.append(
                Message(
                    role="tool",
                    content=result,
                    tool_call_id=p.call.id,
                    name=p.call.name,
                )
            )
        self.info.pending = []
        self.session.notify("tools")

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def run(self) -> str | None:
        """Run the loop; returns the final assistant text (or None)."""
        session = self.session
        self.fsm.transition(State.WAIT)
        rounds = 0
        try:
            return self._run(rounds)
        finally:
            session.last_messages = list(self.messages)

    def _run(self, rounds: int) -> str | None:
        session = self.session
        while rounds < self.max_rounds:
            rounds += 1
            if session.cancel_event.is_set():
                return None
            self._inject_pending_prompts()
            self._update_context_ratio()
            if self._need_compaction():
                session.log(f"compacting context {session.context_ratio:.1%}")
                if self.compact():
                    continue

            try:
                assistant, usage = session.client.chat(
                    self.messages,
                    tools=session.tool_specs() if session.tools_enabled else None,
                    system=self.system,
                    temperature=session.temperature,
                    max_tokens=session.max_tokens,
                    reasoning_effort=session.reasoning_effort,
                    on_delta=session.on_delta,
                )
            except Exception as e:  # noqa: BLE001 - API errors become ERRS
                if session.cancel_event.is_set():
                    return None  # cancelled (Ctrl-C), not an error
                self.info.error = f"Error: {e}"
                session.notify("error")
                break

            session.calibrator.update(usage.input_tokens)
            session.remember_user_text(self.messages)
            session.auto_save(self.messages, self.system)

            if assistant.tool_calls:
                self.info.pending = [
                    PendingToolCall(call=tc) for tc in assistant.tool_calls
                ]
                self.supervisor.reset_nudges()
                self.fsm.transition(State.TOOL)
                self.fsm.transition(State.TRET)
                self._run_tool_round()
                if session.cancel_event.is_set():
                    return None
                self.fsm.transition(State.WAIT)
                continue

            self.fsm.transition()  # WAIT -> TYPE
            self.fsm.transition()  # TYPE -> DONE | ERRS (pending is empty here)
            target = self.fsm.state
            if target in (State.DONE, State.ERRS):
                effective = self.supervisor.supervise(self.fsm, target)
                if effective == State.WAIT:
                    self.messages.append(
                        Message(role="user", content=config.NUDGE_MESSAGE)
                    )
                    continue
                if target == State.ERRS:
                    return f"Error: {self.info.error or 'unknown error'}"
                return assistant.text()

        # round budget exhausted, or an API error broke the loop
        if self.info.error:
            return self.info.error
        return self.messages[-1].text() if self.messages else None


def run_agent_loop(
    session: Any,
    messages: list[Message],
    top_level: bool = True,
    system: str | None = None,
    max_rounds: int = 60,
) -> str | None:
    """Convenience wrapper running a full agent loop."""
    return AgentLoop(
        session, messages=messages, top_level=top_level,
        system=system, max_rounds=max_rounds,
    ).run()
