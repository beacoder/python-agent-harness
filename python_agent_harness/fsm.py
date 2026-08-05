"""Agent finite state machine with completion supervision.

Ported from gptel-agent-harness.el + the upstream gptel FSM:

    INIT -> WAIT -> TYPE -> {ERRS | TOOL -> TRET -> {ERRS | WAIT | DONE} | DONE}

Supervision added by the harness:
- terminal states (DONE/ERRS) on an agentic top-level loop with nudge
  budget left -> inject the nudge message and redirect to WAIT
- nudge counter resets whenever the model makes tool calls
- tool results are sanitized (None -> error placeholder, non-str -> str)
- a failed tool-call batch never strands the FSM (each pending call is
  failed with an error string and the loop continues)
- tool-call processing is idempotent (first result wins)
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import config
from .models import Message, ToolCall


class State(enum.Enum):
    INIT = "INIT"
    WAIT = "WAIT"
    TYPE = "TYPE"
    TOOL = "TOOL"
    TRET = "TRET"
    DONE = "DONE"
    ERRS = "ERRS"
    ABRT = "ABRT"


TERMINAL_STATES = {State.DONE, State.ERRS}

NIL_RESULT_PLACEHOLDER = (
    "Error: tool produced no result (it may have been interrupted or failed to return)."
)

NUDGE_ROLE = "user"


@dataclass
class PendingToolCall:
    call: ToolCall
    spec_name: str = ""


@dataclass
class FsmInfo:
    """Request context carried by the FSM."""

    buffer: object = None  # the session object (TUI session)
    model: str = ""
    backend: str = ""
    tools: bool = False
    top_level: bool = True
    data: dict | None = None
    pending: list[PendingToolCall] = field(default_factory=list)
    error: str | None = None
    history: list[State] = field(default_factory=list)
    harness_injected: bool = False


class Fsm:
    def __init__(
        self,
        info: FsmInfo | None = None,
        handler: Callable[["Fsm", State], None] | None = None,
    ) -> None:
        self.state = State.INIT
        self.info = info or FsmInfo()
        self.handler = handler
        self.supervisor: Optional["Supervisor"] = None

    def transition(self, new_state: State | None = None) -> None:
        target = new_state or self._next()
        self.info.history.append(self.state)
        self.state = target
        if self.handler:
            self.handler(self, target)

    def _next(self) -> State:
        """Resolve the next state via the transition table."""
        table: dict[State, list[tuple[bool, State]]] = {
            State.INIT: [(True, State.WAIT)],
            State.WAIT: [(True, State.TYPE)],
            State.TYPE: [
                (self.info.error is not None, State.ERRS),
                (bool(self.info.pending), State.TOOL),
                (True, State.DONE),
            ],
            State.TOOL: [(True, State.TRET)],
            State.TRET: [
                (self.info.error is not None, State.ERRS),
                (not self.info.pending, State.WAIT),
                (True, State.DONE),
            ],
        }
        for pred, target in table.get(self.state, []):
            if pred:
                return target
        return State.DONE


class Supervisor:
    """Harness supervision layer around FSM transitions."""

    def __init__(self, session: object) -> None:
        self.session = session
        self.nudge_count = 0

    # -- helpers -----------------------------------------------------------
    @property
    def agentic(self) -> bool:
        return bool(self.session.tools_enabled)

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
        supervision redirect returns WAIT.
        """
        self.nudge_count += 1
        return True

    def reset_nudges(self) -> None:
        self.nudge_count = 0

    # -- transition supervision --------------------------------------------
    def supervise(self, fsm: Fsm, target: State | None = None) -> State | None:
        """Return the effective target state, or None to skip the transition.

        Handles:
        - compaction in progress -> let the FSM terminate quietly
        - terminal state on agentic top-level loop -> nudge redirect
        """
        if self.session.compacting:
            return target
        if target in TERMINAL_STATES:
            if (
                self.agentic
                and fsm.info.top_level
                and self.can_nudge()
                and not fsm.info.pending
            ):
                if self.inject_nudge():
                    return State.WAIT
        return target


class ToolExecutionError(Exception):
    """Raised when a tool call batch fails wholesale."""


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


def fail_pending_tool_calls(fsm: Fsm, error: BaseException) -> None:
    """Fail every pending tool call with an error string.

    Mirrors gptel-agent-harness-fsm--fail-pending-tool-calls: each
    pending call without a result gets an error result so the caller
    can deliver tool messages and the FSM can move TOOL -> TRET.
    Results are set in place; the loop drains the list afterwards.
    """
    for pending in fsm.info.pending:
        if pending.call.result is None:
            pending.call.result = f"Error: tool call failed — {error}"
