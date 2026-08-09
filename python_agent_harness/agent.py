"""Core agent loop: send -> tool -> supervise.

Implements the gptel-agent-harness supervision semantics:

- context ratio is computed before each top-level request; compaction is
  triggered past the threshold (top-level, agentic, not already compacting)
- queued plan/build-mode prompts are injected before sending
- a terminal response on an agentic top-level loop nudges the model back
  to work while nudge budget remains (max 2), reset on tool calls
- tool results are sanitized (None -> error placeholder, non-str -> str)
- tool-call batches never strand the loop: failures become error results
- every tool call in a round runs concurrently in a thread pool (results
  delivered in original order); interactive prompts stay serialized
- token calibration is updated from API-reported input tokens
- sessions are auto-saved after each response
- a cancelled run with no successor salvages its partial history
  (truncated to the last complete tool round) instead of losing it;
  a stale worker superseded by a newer run never touches shared state
"""

from __future__ import annotations

import json
from typing import Any

from . import config
from .prompts import last_user_request, read_prompt_file
from .models import Message, ToolCall
from .token_estimator import context_window_for, estimate_payload_tokens


class AgentLoop:
    """Runs one agent session until terminal (main) or max rounds (sub-agent)."""

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
        # max_rounds only bounds sub-agent loops: the main agent runs until
        # the model gives a terminal response or the user aborts it (Ctrl-C)
        self.max_rounds = max_rounds if not top_level else None
        self.pending: list[ToolCall] = []
        self.error: str | None = None
        self.harness_injected: bool = False
        self.supervisor = Supervisor(session)
        # Cancellation identity for this run: cancel() bumps the session
        # generation, so a stale worker from a cancelled run stays
        # cancelled even after the next run clears the shared event (and
        # must not touch shared state).  Captured at construction — the
        # worker thread starts right after, and a run superseded between
        # construction and start must not adopt the new generation.
        self._cancel_gen = session.cancel_generation
        # Run identity for this run: a newer top-level run bumps
        # `session.run_generation`, marking this worker stale —
        # superseded, so it must never touch shared state.  Distinct
        # from cancellation: a cancelled run with no successor still
        # owns the session and may salvage its partial history.
        self._run_gen = session.run_generation

    def _is_cancelled(self) -> bool:
        """Whether THIS run must stop (cancelled or superseded).

        The plain event is not enough: `_start_agent` clears it before
        every run, so a worker from a cancelled run that finishes late
        (e.g. after a long tool call) would otherwise see it cleared and
        clobber the new run's `session.last_messages`.  A superseded
        worker (a newer run bumped `run_generation`) is dead too: it
        must stop working and must not touch shared state.
        """
        return (
            self.session.cancel_event.is_set()
            or self.session.cancel_generation != self._cancel_gen
            or self.session.run_generation != self._run_gen
        )

    def _is_stale(self) -> bool:
        """Whether a newer top-level run owns the session.

        Distinct from cancelled: a cancelled run with no successor still
        owns the session and may salvage its partial history; a stale
        worker must never touch shared state (its partial history would
        clobber the new run's).
        """
        return self.session.run_generation != self._run_gen

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
                    Message(
                        role="user",
                        content=self.session.plan_mode.plan_reminder(),
                        injected=True,
                    ),
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
            self.messages.insert(
                insert_at + i, Message(role="user", content=text, injected=True)
            )

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
        if not self.top_level and call.name in config.SUBAGENT_EXCLUDED_TOOLS:
            # defense in depth: a hallucinated call must never reach the
            # registry — the spec was filtered, so refuse it here too
            return (
                f"Error: {call.name} is not available to sub-agents — "
                "it is a parent-only tool"
            )
        args = call.arguments
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if not isinstance(args, dict):
            args = {}
        return self.session.execute_tool(call.name, args, call_id=call.id)

    def _deliver_tool_result(self, p: ToolCall, result: str) -> None:
        """Append one tool result message for call P (parent thread only)."""
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

    def _run_tools_parallel(
        self, calls: list[ToolCall], results: dict[str, str]
    ) -> None:
        """Run CALLS concurrently in a thread pool, filling RESULTS.

        Each call executes in its own worker thread: the session's
        shared state is concurrency-safe (thread-local diff slots,
        serialized interactive prompts), so every
        tool issued in the same round — Agent calls included, whose
        sub-agents are isolated by design — runs in parallel.  Delivery
        happens later, in original tool-call order, by the parent
        thread.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def run_one(p: ToolCall) -> str:
            # A cancel landing while the task is still QUEUED must skip
            # it (tools have side effects): the sequential loop used to
            # check before every call, so keep that guarantee — a
            # task already RUNNING cannot be stopped, but one that has
            # not started yet must not run after Ctrl-C.
            if self._is_cancelled():
                return "Error: tool call cancelled (user aborted the run)."
            return self._execute_tool_call(p)

        with ThreadPoolExecutor(
            max_workers=min(len(calls), config.PARALLEL_TOOL_MAX),
            thread_name_prefix="tool",
        ) as pool:
            futures = {pool.submit(run_one, p): p for p in calls}
            for fut in as_completed(futures):
                p = futures[fut]
                try:
                    results[p.id] = sanitize_tool_result(fut.result())
                except Exception as e:  # noqa: BLE001 - containment boundary
                    results[p.id] = (
                        f"Error: tool {p.name!r} crashed in a worker "
                        f"thread — {e}"
                    )

    def _run_tool_round(self) -> None:
        """Execute all pending tool calls concurrently; deliver results.

        The assistant message carrying the tool calls was already
        appended by the main loop; here we add the per-call results.

        All tools issued in the round run CONCURRENTLY in a thread
        pool — the session's shared state is concurrency-safe
        (thread-local diff slots, serialized interactive prompts).
        Results are delivered in the original tool-call order
        regardless of execution order.
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
        results: dict[str, str] = {}
        self._run_tools_parallel(pending, results)
        if self._is_cancelled():
            # cancelled mid-round: tools already submitted may have run
            # (their side effects are done), but the results stay local
            # to this (dead) run
            self.pending = []
            return
        for p in pending:
            if self._is_cancelled():
                self.pending = []
                return
            self._deliver_tool_result(p, results[p.id])
        self.pending = []
        self.session.notify("tools")

    def _salvage_messages(self) -> list[Message]:
        """Longest valid prefix of ``self.messages`` for the shared history.

        A cancelled run may end mid-tool-round: the assistant message
        carrying the tool calls is present but some (or all) results are
        missing.  Committing that as-is would hand the next turn an
        invalid request (a tool call without its response), so cut back
        to the last complete round — the model redoes the dangling work
        on the next turn.
        """
        msgs = self.messages
        open_round: int | None = None
        pending: dict[str, bool] = {}
        for i, m in enumerate(msgs):
            if m.role == "assistant":
                if m.tool_calls:
                    if open_round is not None:
                        return msgs[:open_round]
                    open_round = i
                    pending = {tc.id: False for tc in m.tool_calls}
                elif open_round is not None:
                    return msgs[:open_round]
            elif m.role == "tool":
                if m.tool_call_id in pending:
                    pending[m.tool_call_id] = True
                    if all(pending.values()):
                        open_round = None
                        pending = {}
            elif open_round is not None:
                return msgs[:open_round]
        if open_round is not None:
            return msgs[:open_round]
        return msgs

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def run(self) -> str | None:
        """Run the loop; returns the final assistant text (or None)."""
        session = self.session
        rounds = 0
        try:
            return self._run(rounds)
        finally:
            # A stale worker (a newer run has started) must never touch
            # shared state, and a sub-agent must never overwrite the
            # parent's history.  A merely cancelled run still owns the
            # session, though: commit its partial history (truncated to
            # the last complete tool round) so the interrupted turn is
            # not lost — the next turn resumes from it instead of
            # re-asking.
            if not self._is_stale() and self.top_level:
                salvaged = self._salvage_messages()
                session.last_messages = list(salvaged)
                if self._is_cancelled():
                    # Persist the partial turn now: auto-save only runs
                    # after successful responses, so without this the
                    # interrupted turn would never reach the session
                    # file — the next turn's save would overwrite it
                    # without ever containing it.
                    session.auto_save(salvaged, self.system)
                    # Elisp parity: the title is generated on the first
                    # save, not only on clean completion — an
                    # interrupted session still gets a meaningful name
                    # (one-shot; no-op when already titled/pending).
                    session.generate_session_title()
                else:
                    # Loop finished: give the session a meaningful title
                    # from the first real user message (one-shot; no-op
                    # when the title already exists or generation is in
                    # flight)
                    session.generate_session_title()

    def _run(self, rounds: int) -> str | None:
        session = self.session
        while self.max_rounds is None or rounds < self.max_rounds:
            rounds += 1
            if self._is_cancelled():
                return None
            self._inject_pending_prompts()
            if self.top_level:
                # sub-agents must not touch the shared context accounting:
                # their payload (fresh context) is structurally different,
                # so their ratio/usage would skew the parent's
                self._update_context_ratio()
                if self._need_compaction():
                    session.log(f"compacting context {session.context_ratio:.1%}")
                    if self.compact():
                        continue

            def safe_delta(text: str) -> None:
                if not self._is_cancelled() and session.on_delta is not None:
                    session.on_delta(text)

            try:
                # sub-agents are one-shot tasks: they must not see (or
                # call) parent-only tools — Agent (no nesting), Question
                # and PlanExit (interactive/handoff), TodoWrite (the
                # parent's own progress tracking) — filtered from the
                # specs before sending
                tools = session.tool_specs(
                    exclude=config.SUBAGENT_EXCLUDED_TOOLS if not self.top_level else ()
                )
                assistant, usage = session.client.chat(
                    self.messages,
                    tools=tools if session.tools_enabled else None,
                    system=self.system,
                    temperature=session.temperature,
                    max_tokens=session.max_tokens,
                    reasoning_effort=session.reasoning_effort,
                    stream=session.stream,
                    # sub-agents must not stream into the parent's live
                    # stream row — their text is private until returned
                    on_delta=(safe_delta if self.top_level else None),
                    # poll cancellation during retry backoff so Ctrl-C
                    # aborts promptly instead of after the full sleep
                    cancel_check=self._is_cancelled,
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

            if self.top_level:
                session.calibrator.update(usage.input_tokens)
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
                    Message(role="user", content=config.NUDGE_MESSAGE, injected=True)
                )
                continue
            if self.error:
                return f"Error: {self.error or 'unknown error'}"
            return assistant.text()

        # round budget exhausted (sub-agents only), or an API error broke
        # the loop
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
