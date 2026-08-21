"""Tool execution for the agent FSM: run/deliver tool calls, salvage history.

Extracted from agent.py (no logic changes): the FSM driver delegates
tool plumbing to ``ToolRunner``, which reads/writes the loop's shared
state (``messages`` / ``pending`` / ``info``) through the loop
reference.  Tool calls are always executed and delivered via the
loop's ``_execute_tool_call`` / ``_deliver_tool_result`` methods so
subclass or test overrides of those methods keep working.
"""

from __future__ import annotations

import json
import time
from typing import Any

from . import config
from .models import Message, ToolCall
from .tools.base import PendingToolResult

NIL_RESULT_PLACEHOLDER = (
    "Error: tool produced no result (it may have been interrupted or failed to return)."
)


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


class ToolRunner:
    """Runs and delivers tool calls for one agent loop.

    Mirrors gptel's `gptel--handle-tool-use': synchronous tools
    (Read, Edit, Glob, ...) execute ONE AT A TIME, in call order;
    asynchronous tools (Bash, Agent — those whose ``run`` returns a
    ``PendingToolResult``) are dispatched in line and run concurrently
    in the background, their results awaited afterwards, again in
    original call order.
    """

    def __init__(self, loop: Any) -> None:
        self.loop = loop

    def execute_tool_call(self, call: ToolCall) -> str | PendingToolResult:
        loop = self.loop
        if not loop.top_level and call.name in config.SUBAGENT_EXCLUDED_TOOLS:
            # defense in depth: a hallucinated call must never reach the
            # registry — the spec was filtered, so refuse it here too
            return f"Error: {call.name} is not available to sub-agents — it is a parent-only tool"
        args = call.arguments
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if not isinstance(args, dict):
            args = {}
        # Validate required parameters from the tool schema
        tool = loop.session.registry.get(call.name)
        if tool is not None:
            required = tool.parameters.get("required", [])
            missing = [k for k in required if k not in args]
            if missing:
                return f"Error: {call.name} is missing required argument(s): {', '.join(missing)}"
        return loop.session.execute_tool(call.name, args, call_id=call.id)

    def deliver_tool_result(self, p: ToolCall, result: str) -> None:
        """Append one tool result message for call P (parent thread only)."""
        loop = self.loop
        p.result = result
        if hasattr(loop.session, "take_diff"):
            p.diff = loop.session.take_diff(p.id)
        loop.messages.append(
            Message(
                role="tool",
                content=result,
                tool_call_id=p.id,
                name=p.name,
            )
        )
        if loop.top_level and not loop._is_cancelled():
            # only the top-level loop mirrors its messages onto the
            # shared session: a sub-agent runs inside the parent's
            # tool round and must never clobber the parent's
            # conversation history (the TUI renders from it)
            loop.session.last_messages = list(loop.messages)

    def run_tools(self, calls: list[ToolCall], results: dict[str, str]) -> None:
        """Run CALLS in model-emitted order, filling RESULTS.

        Mirrors gptel's `gptel--handle-tool-use': synchronous tools
        (Read, Edit, Glob, ...) execute ONE AT A TIME, in call order;
        asynchronous tools (Bash, Agent — those whose ``run`` returns a
        ``PendingToolResult``) are dispatched in line and run
        concurrently in the background, their results awaited
        afterwards, again in original call order.  Delivery happens
        later, in original tool-call order, by the caller.

        A cancel landing before a call starts skips it (tools have side
        effects); a call already running — or an async tool already
        dispatched — cannot be stopped, but its result stays local to
        the (dead) run.
        """
        loop = self.loop
        async_calls: list[tuple[ToolCall, PendingToolResult]] = []
        for p in calls:
            # A cancel landing while a call is still QUEUED must skip
            # it (tools have side effects): the sequential loop checks
            # before every call, so a call that has not started yet must
            # not run after Ctrl-C.
            if loop._is_cancelled():
                results[p.id] = "Error: tool call cancelled (user aborted the run)."
                continue
            # Notify the TUI which tool is currently executing so the
            # status bar can show the active tool name beside the spinner.
            if loop.top_level:
                loop.session.notify("tool_running", p.name)
            start = time.monotonic()
            try:
                result = loop._execute_tool_call(p)
            except Exception as e:  # noqa: BLE001 - containment boundary
                p.elapsed = time.monotonic() - start
                results[p.id] = f"Error: tool {p.name!r} crashed during execution — {e}"
                continue
            if isinstance(result, PendingToolResult):
                # async tool (e.g. Bash): run() spawned the work and
                # returned its handle immediately; await the real result
                # after the sequential loop so sibling calls keep
                # executing in the meantime
                async_calls.append((p, result))
            else:
                p.elapsed = time.monotonic() - start
                results[p.id] = sanitize_tool_result(result)
        for p, pending in async_calls:
            start = time.monotonic()
            try:
                result = pending.wait()
            except Exception as e:  # noqa: BLE001 - containment boundary
                results[p.id] = f"Error: tool {p.name!r} crashed during execution — {e}"
            else:
                p.elapsed = time.monotonic() - start
                results[p.id] = sanitize_tool_result(result)

    def execute_pending(self) -> None:
        """TOOL state: run the round's pending tool calls.

        The assistant message carrying the tool calls was already
        appended by the WAIT state.  Results land in
        ``self.info["tool_result"]`` and are delivered by the TRET
        state in original tool-call order.

        Synchronous tools run ONE AT A TIME in model-emitted order
        (gptel-style); asynchronous tools (Bash, Agent) are dispatched
        in line and run concurrently in the background.
        """
        loop = self.loop
        pending = list(loop.pending)
        if not pending:
            return
        if loop._is_cancelled():
            # Ctrl-C before the round started: do not run tools (they
            # have side effects) and do not touch shared state — a stale
            # worker must never mirror its partial history over the next
            # run's `session.last_messages`.
            return
        results: dict[str, str] = {}
        loop._run_tools(pending, results)
        if loop._is_cancelled():
            # cancelled mid-round: tools already submitted may have run
            # (their side effects are done), but the results stay local
            # to this (dead) run
            loop.pending = []
            return
        loop.info["tool_result"] = results

    def deliver_results(self) -> None:
        """TRET state: deliver the round's results to the conversation.

        Results are appended as tool messages in the original
        tool-call order regardless of execution order.  On cancel the
        partial delivery is discarded — the shared history salvage
        cuts the dangling round so no tool call is left unanswered.
        """
        loop = self.loop
        pending = list(loop.pending)
        if not pending:
            return
        if loop._is_cancelled():
            loop.pending = []
            return
        results = loop.info.get("tool_result", {})
        for p in pending:
            if loop._is_cancelled():
                loop.pending = []
                return
            loop._deliver_tool_result(p, results[p.id])
            # Notify per-tool so the TUI rebuilds history progressively
            # (each tool result appears as soon as it is committed to
            # the conversation, instead of all at once at the end)
            if loop.top_level:
                loop.session.notify("tools")
        loop.pending = []

    def salvage_messages(self) -> list[Message]:
        """Longest valid prefix of ``self.messages`` for the shared history.

        A cancelled run may end mid-tool-round: the assistant message
        carrying the tool calls is present but some (or all) results are
        missing.  Committing that as-is would hand the next turn an
        invalid request (a tool call without its response), so cut back
        to the last complete round — the model redoes the dangling work
        on the next turn.
        """
        loop = self.loop
        msgs = loop.messages
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
