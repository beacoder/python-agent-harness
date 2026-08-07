"""Core agent loop: send -> tool -> supervise.

Implements the gptel-agent-harness supervision semantics:

- context ratio is computed before each top-level request; compaction is
  triggered past the threshold (top-level, agentic, not already compacting)
- queued plan/build-mode prompts are injected before sending
- a terminal response on an agentic top-level loop nudges the model back
  to work while nudge budget remains (max 2), reset on tool calls
- tool results are sanitized (None -> error placeholder, non-str -> str)
- tool-call batches never strand the loop: failures become error results
- token calibration is updated from API-reported input tokens
- sessions are auto-saved after each response
"""

from __future__ import annotations

import json
from typing import Any

from . import config
from .compaction import last_user_request, read_prompt_file
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
        # fall back to the session's prompt so a run never loses it;
        # sub-agent loops use the session's SUB-AGENT prompt (their own),
        # never the parent's system prompt (which carries the parent's
        # context and task-completion rules)
        if system is not None:
            self.system = system
        else:
            attr = "system_prompt" if top_level else "subagent_system_prompt"
            self.system = getattr(session, attr, None)
        self.max_rounds = max_rounds
        self.pending: list[ToolCall] = []
        self.error: str | None = None
        self.harness_injected: bool = False
        self.supervisor = Supervisor(session)
        # Cancellation identity for this run, captured at run() start:
        # cancel() bumps the session generation, so a stale worker from
        # a cancelled run stays cancelled even after the next run clears
        # the shared event (and must not touch shared state).
        self._cancel_gen = 0

    def _is_cancelled(self) -> bool:
        """Whether THIS run was cancelled (event set or generation moved).

        The plain event is not enough: `_start_agent` clears it before
        every run, so a worker from a cancelled run that finishes late
        (e.g. after a long tool call) would otherwise see it cleared and
        clobber the new run's `session.last_messages`.
        """
        return (
            self.session.cancel_event.is_set()
            or self.session.cancel_generation != self._cancel_gen
        )

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
            if self.session.plan_mode.is_plan and not self.harness_injected:
                self.messages.insert(
                    len(self.messages),
                    Message(role="user", content=self.session.plan_mode.plan_reminder()),
                )
                self.harness_injected = True
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
        """Execute all pending tool calls; deliver results as messages.

        The assistant message carrying the tool calls was already
        appended by the main loop; here we add the per-call results.
        """
        pending = list(self.pending)
        if not pending:
            return
        if self._is_cancelled():
            # Ctrl-C before the round started: do not run tools (they
            # have side effects) and do not touch shared state — a stale
            # worker must never mirror its partial history over the next
            # run's `session.last_messages`.
            return
        for p in pending:
            if self._is_cancelled():
                # cancelled mid-round: stop running further tools; the
                # results already delivered stay local to this (dead) run
                self.pending = []
                return
            result = sanitize_tool_result(self._execute_tool_call(p))
            p.result = result
            if hasattr(self.session, "take_diff"):
                p.diff = self.session.take_diff(p.id)
            self.messages.append(
                Message(
                    role="tool",
                    content=result,
                    tool_call_id=p.id,
                    name=p.name,
                )
            )
            if self.top_level and not self._is_cancelled():
                # only the top-level loop mirrors its messages onto the
                # shared session: a sub-agent runs inside the parent's
                # tool round and must never clobber the parent's
                # conversation history (the TUI renders from it)
                self.session.last_messages = list(self.messages)
        self.pending = []
        self.session.notify("tools")

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def run(self) -> str | None:
        """Run the loop; returns the final assistant text (or None)."""
        session = self.session
        self._cancel_gen = session.cancel_generation
        rounds = 0
        try:
            return self._run(rounds)
        finally:
            # A cancelled run must not clobber state for the next run,
            # and a sub-agent must never overwrite the parent's history.
            if not self._is_cancelled() and self.top_level:
                session.last_messages = list(self.messages)
                # Loop finished: give the session a meaningful title from
                # the first real user message (one-shot; no-op when the
                # title already exists or generation is in flight)
                session.generate_session_title()

    def _run(self, rounds: int) -> str | None:
        session = self.session
        while rounds < self.max_rounds:
            rounds += 1
            if self._is_cancelled():
                return None
            self._inject_pending_prompts()
            self._update_context_ratio()
            if self._need_compaction():
                session.log(f"compacting context {session.context_ratio:.1%}")
                if self.compact():
                    continue

            def safe_delta(text: str) -> None:
                if not self._is_cancelled() and session.on_delta is not None:
                    session.on_delta(text)

            try:
                assistant, usage = session.client.chat(
                    self.messages,
                    tools=session.tool_specs() if session.tools_enabled else None,
                    system=self.system,
                    temperature=session.temperature,
                    max_tokens=session.max_tokens,
                    reasoning_effort=session.reasoning_effort,
                    # sub-agents must not stream into the parent's live
                    # stream row — their text is private until returned
                    on_delta=(safe_delta if self.top_level else None),
                )
            except Exception as e:  # noqa: BLE001 - API errors become ERRS
                if self._is_cancelled():
                    return None  # cancelled (Ctrl-C), not an error
                self.error = f"Error: {e}"
                session.notify("error")
                break

            if self._is_cancelled():
                return None  # response arrived after cancel: drop it

            # persist the assistant response in the conversation history
            # (text and/or tool calls) so later turns and the UI see it
            if assistant.text().strip() or assistant.tool_calls:
                self.messages.append(assistant)
                if not assistant.tool_calls and self.top_level:
                    session.last_messages = list(self.messages)

            session.calibrator.update(usage.input_tokens)
            if self.top_level:
                session.remember_user_text(self.messages)
                session.auto_save(self.messages, self.system)

            if assistant.tool_calls:
                self.pending = list(assistant.tool_calls)
                self.supervisor.reset_nudges()
                self._run_tool_round()
                if self._is_cancelled():
                    return None
                continue

            # no tool calls: decide whether to nudge or stop
            if self.supervisor.supervise(
                terminal=True,
                agentic=bool(session.tools_enabled),
                top_level=self.top_level,
                pending=bool(self.pending),
            ):
                self.messages.append(
                    Message(role="user", content=config.NUDGE_MESSAGE)
                )
                continue
            if self.error:
                return f"Error: {self.error or 'unknown error'}"
            return assistant.text()

        # round budget exhausted, or an API error broke the loop
        if self.error:
            return self.error
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
NIL_RESULT_PLACEHOLDER = (
    "Error: tool produced no result (it may have been interrupted or failed to return)."
)


class Supervisor:
    """Completion supervision: nudge the model when it stops too early."""

    def __init__(self, session: object) -> None:
        self.session = session
        self.nudge_count = 0

    # -- helpers -----------------------------------------------------------
    @property
    def alive(self) -> bool:
        return self.session.alive

    def can_nudge(self) -> bool:
        """Fails closed: a dead session has NO nudge budget.

        Mirrors the Emacs logic where a dead buffer can never record
        nudges and would otherwise loop forever.
        """
        return self.alive and self.nudge_count < config.MAX_NUDGES

    def inject_nudge(self) -> bool:
        """Increment the nudge counter. Returns True on success; never raises.

        The caller (agent loop) appends the nudge message itself when the
        supervision decides to nudge.
        """
        self.nudge_count += 1
        return True

    def reset_nudges(self) -> None:
        self.nudge_count = 0

    # -- supervision -------------------------------------------------------
    def supervise(
        self,
        *,
        terminal: bool,
        agentic: bool,
        top_level: bool,
        pending: bool,
    ) -> bool:
        """Decide whether to nudge the model back to work.

        Returns True to inject a nudge and run another round; False to
        let the loop terminate.

        Handles:
        - compaction in progress -> never nudge, let the loop terminate
        - terminal response on an agentic top-level loop with nudge
          budget left and no pending tool calls -> nudge
        """
        if self.session.compacting:
            return False
        if terminal and agentic and top_level and self.can_nudge() and not pending:
            self.inject_nudge()
            return True
        return False


def sanitize_tool_result(result: object) -> str:
    """Sanitize a tool result for the model.

    - str (incl. empty) kept as-is
    - None -> error placeholder (backends reject JSON null content)
    - anything else -> str()
    """
    if result is None:
        return NIL_RESULT_PLACEHOLDER
    if isinstance(result, str):
        return result
    return str(result)
