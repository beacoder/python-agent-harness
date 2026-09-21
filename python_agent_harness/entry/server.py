"""Resident agent server: a persistent, bidirectional runtime boundary.

Where ``headless --json`` is one prompt per process over a write-only
JSONL pipe, ``serve`` keeps a live ``Controller``/``Session`` resident
and speaks a request/response JSONL protocol over stdin/stdout — the
same subprocess boundary as headless (containerizable, trust boundary
preserved), but the host can:

- submit multiple prompts over the process lifetime (no per-turn
  interpreter spawn), with conversation history retained between them
  (multi-turn memory for free);
- receive ``ask`` events and deliver answers mid-run (a web user can
  respond to the agent's Question tool / PlanExit confirm), which the
  one-shot headless protocol structurally cannot do (it auto-answers);
- cancel via a protocol message (no signal semantics).

Protocol (one JSON object per line), host → agent:
  {"op": "submit", "prompt": ..., "run_id": ...}
  {"op": "answer", "run_id": ..., "answers": [...]}
  {"op": "cancel", "run_id": ...}
  {"op": "ping"} / {"op": "shutdown"}
agent → host (one JSON object per line, ``seq`` on every line):
  {"type": "ready", ...}                       first line, no run_id
  {"type": "start"|"delta"|"notify"|"log", "run_id": ...}
  {"type": "result", "run_id": ..., "answer": ..., "errors": [...], ...}
  {"type": "ack"|"pong"|"error", ...}          control lines, no run_id
  notify with kind "ask" carries {"kind": "ask"|"confirm", ...} data

Concurrency: one run at a time (the web controller already enforces
"one running run per conversation"); a submit while a run is active is
rejected with an ``error`` line.  The reader loop stays live while a
run executes — ops (answer/cancel/ping/shutdown) are processed on the
reader thread while the run's own thread waits on the agent worker —
so a host can answer a pending ask or cancel mid-run.  Events from the
agent's worker thread are handed to the caller through the view (whose
lock serializes the pre-start buffer against live emission), so lines
on the wire stay strictly ordered by ``seq``.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from typing import Any, TextIO

from ..io.text_filter import strip_final_check
from ..session.session import Session
from .controller import Controller
from .headless import JsonlView

DEFAULT_ANSWER_TIMEOUT = 0.0  # wait forever for a host answer


class _AskState:
    """One pending interactive prompt awaiting a host answer.

    ``ServerView.confirm``/``ask`` (agent worker thread) create the
    state, emit the ask line, and block on ``wait``; the server's
    reader resolves it via ``ServerView.deliver``/``resolve_confirm``.
    """

    def __init__(self, run_id: str | None, kind: str) -> None:
        self.run_id = run_id
        self.kind = kind  # "ask" | "confirm"
        self.event = threading.Event()
        self.answer: str = "Unanswered"
        self.created = time.monotonic()


class ServerView(JsonlView):
    """``JsonlView`` extended with resolvable ask/confirm.

    The run's live event flow (start/delta/notify/log) reuses the
    JsonlView machinery wholesale — seq numbering, run_id echo, the
    pre-start buffer, the write lock.  ``confirm``/``ask`` replace the
    headless auto-answer: publish an ``ask`` notify, then block the
    worker thread until ``deliver``/``resolve_confirm`` resolves it,
    the deadline passes, or the run is cancelled (a cancelled run must
    unblock the worker at once — the agent loop is unwinding anyway).
    """

    def __init__(
        self,
        out: TextIO | None = None,
        err: TextIO | None = None,
        run_id: str | None = None,
        answer_timeout: float = DEFAULT_ANSWER_TIMEOUT,
    ) -> None:
        super().__init__(out=out, err=err, run_id=run_id)
        # The active pending interactive prompt, if any.  Published by
        # the worker thread (inside the session's _interactive_lock),
        # resolved by the reader thread; guarded by its own lock.
        self._pending: _AskState | None = None
        self._pending_lock = threading.Lock()
        # Seconds to wait for a host answer before falling back to
        # "Unanswered" (headless semantics).  0 = wait forever.
        self.answer_timeout = answer_timeout
        # Set when the run is cancelled: a pending ask must unblock at
        # once (the agent loop is unwinding anyway).
        self._cancelled = threading.Event()

    # -- interactive prompts ---------------------------------------------------

    def _wait_answer(self, state: _AskState) -> str:
        """Block until answered, cancelled, or the deadline passes."""
        deadline = time.monotonic() + self.answer_timeout if self.answer_timeout > 0 else None
        while True:
            if state.event.wait(0.1):
                # cancellation outranks a concurrent resolve (the run is
                # unwinding; the agent must see an empty answer, TUI
                # parity) — cancel may land while we are inside wait()
                return "" if self._cancelled.is_set() else state.answer
            if self._cancelled.is_set():
                return ""
            if deadline is not None and time.monotonic() >= deadline:
                return "Unanswered"

    def answer(self, answers: list[str]) -> bool:
        """Resolve the pending ask/confirm with *answers* (host-side).

        Routes by the pending state's kind: confirm takes the first
        answer verbatim (the caller decides yes/no), ask joins multiple
        values with ", " (mirroring the TUI's multi-select resolution).
        Returns False when nothing is pending.
        """
        with self._pending_lock:
            state = self._pending
            if state is None:
                return False
            self._pending = None
        if state.kind == "confirm":
            state.answer = answers[0]
        elif len(answers) > 1:
            state.answer = ", ".join(a.strip() for a in answers if a.strip())
        else:
            state.answer = answers[0]
        state.event.set()
        return True

    def cancel_pending(self) -> None:
        """Unblock a pending interactive prompt (run cancelled)."""
        self._cancelled.set()
        with self._pending_lock:
            state = self._pending
            self._pending = None
        if state is not None:
            state.event.set()

    def reset_for_run(self, run_id: str | None) -> None:
        """Re-arm the view for a new run: fresh seq counter, new run_id."""
        with self._lock:
            self._buffer = []
            self._seq = 0
        self._cancelled.clear()
        self.run_id = run_id

    # -- HeadlessView overrides -------------------------------------------------

    def confirm(self, prompt: str) -> bool:
        state = _AskState(self.run_id, "confirm")
        with self._pending_lock:
            self._pending = state
        self._emit(
            {
                "type": "notify",
                "kind": "ask",
                "data": {"kind": "confirm", "prompt": prompt},
            }
        )
        answer = self._wait_answer(state)
        return answer.strip().lower() in ("y", "yes", "true", "1")

    def ask(self, questions: list[dict]) -> str:
        state = _AskState(self.run_id, "ask")
        with self._pending_lock:
            self._pending = state
        self._emit(
            {"type": "notify", "kind": "ask", "data": {"kind": "ask", "questions": questions}}
        )
        return self._wait_answer(state)

    # -- test seam ------------------------------------------------------------

    def take_pending(self) -> _AskState | None:
        """Detach the pending ask state (test inspection hook)."""
        with self._pending_lock:
            state = self._pending
            self._pending = None
        return state


class AgentServer:
    """Request loop over stdin/stdout driving a resident Controller.

    Reads host ops from *inp* (one JSON line each) and writes protocol
    lines to *out*.  The resident ``Controller`` is created with the
    session; a ``ServerView`` is attached when a run starts and re-armed
    per run (fresh seq/run_id), so every run's lines are numbered from 1
    under its own correlation id.

    Exactly one run may be active; ``submit`` while active is rejected
    with ``error``.  Answers/cancels must name the active run; stale
    ones are rejected.  Protocol-level failures (unknown op, malformed
    line, answer with nothing pending) emit ``error`` lines; run
    failures surface on the ``result`` line's ``errors``/``cancelled``
    fields, exactly like headless.
    """

    def __init__(
        self,
        session: Session,
        inp: TextIO,
        out: TextIO,
        err: TextIO | None = None,
        answer_timeout: float = DEFAULT_ANSWER_TIMEOUT,
    ) -> None:
        self.session = session
        self.controller = Controller(session)
        self.inp = inp
        self.out = out
        self.err = err if err is not None else sys.stderr
        self.answer_timeout = answer_timeout
        self.view: ServerView | None = None
        self._active_run_id: str | None = None
        self._active_guard = threading.Lock()  # active-run transitions
        self._stopped = threading.Event()

    # -- outbound lines ---------------------------------------------------------

    def _write_line(self, payload: dict[str, Any]) -> None:
        """Write one protocol line, serialized against run-thread output.

        Live events (start/delta/notify/log/result) are written by the
        run's threads under the attached view's ``_lock``; control
        lines (ready/pong/error) come from the reader thread.  Taking
        the same lock here keeps every line whole — without it an
        ``error`` emitted mid-run could interleave with a streamed
        event line on the wire.
        """
        view = self.view
        if view is not None:
            with view._lock:
                self.out.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
                self.out.flush()
        else:
            self.out.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            self.out.flush()

    def _error(self, message: str) -> None:
        self._write_line({"type": "error", "error": message})

    # -- run execution ------------------------------------------------------------

    def _cancelled_now(self) -> bool:
        event = getattr(self.session, "cancel_event", None)
        return event is not None and event.is_set() is True

    def _final_answer(self) -> str:
        """The run's final assistant answer, TUI-filtered (headless shape)."""
        for msg in reversed(self.session.last_messages):
            if getattr(msg, "role", None) != "assistant":
                continue
            return strip_final_check(msg.text_without_reasoning()).strip()
        return ""

    def _usage_snapshot(self) -> dict[str, Any] | None:
        """Cumulative input/output tokens and rounds of the current run."""
        totals = getattr(self.session, "usage_totals", None)
        if not isinstance(totals, dict):
            return None
        usage: dict[str, Any] = {"input": 0, "output": 0, "rounds": 0}
        with totals.get("_lock", threading.Lock()):
            for key in ("input", "output", "rounds"):
                value = totals.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    usage[key] = int(value)
        return usage

    def _execute_run(self, prompt: str, run_id: str) -> None:
        """Run one prompt to completion, streaming lines as they happen.

        Runs on its own thread (spawned by ``op_submit``): the agent's
        worker thread emits live events through the view while this
        thread waits on the worker, then writes the final result line.
        The reader loop stays live meanwhile, so answer/cancel ops can
        be processed mid-run.
        """
        view = ServerView(
            out=self.out,
            err=self.err,
            run_id=run_id,
            answer_timeout=self.answer_timeout,
        )
        self.view = view
        self.controller.attach_view(view)
        handle = self.controller.submit(prompt)
        usage = self._usage_snapshot()
        model = self.session.model
        if handle is None:
            # Failure before start, headless contract: the result line is
            # the only line of the run (seq 1, run_id echoed).
            view.emit_result(
                "",
                errors=["nothing to send"],
                usage=usage,
                model=model,
            )
            return
        # Flush events the worker emitted between submit() and now (the
        # emit_start contract: start is guaranteed to be the first line).
        view.emit_start(prompt, list(handle.warnings))
        handle.worker.join()
        cancelled = self._cancelled_now()
        errors = list(view.errors)
        if not errors and not cancelled and not self._final_answer():
            # The worker finished without an assistant message and without
            # surfacing an error (e.g. the agent loop died before any
            # output): the result line must carry a reason (headless does
            # this via its exit code).
            errors.append("run produced no answer")
        view.cancel_pending()
        view.emit_result(
            self._final_answer(),
            usage=usage,
            model=model,
            cancelled=cancelled,
            errors=errors,
        )

    # -- op handlers ----------------------------------------------------------------

    def op_submit(self, op: dict[str, Any]) -> None:
        run_id = str(op.get("run_id") or "")
        if not run_id:
            self._error("submit requires run_id")
            return
        prompt = op.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            self._error("submit requires a non-empty prompt")
            return
        with self._active_guard:
            if self._active_run_id is not None:
                self._error("a run is already active")
                return
            self._active_run_id = run_id
        thread = threading.Thread(
            target=self._run_thread, args=(prompt, run_id), name=f"serve-run-{run_id}"
        )
        thread.start()

    def _run_thread(self, prompt: str, run_id: str) -> None:
        try:
            self._execute_run(prompt, run_id)
        except Exception as e:  # noqa: BLE001 - a run failure must not kill the server
            self._error(f"run failed: {e}")
        finally:
            with self._active_guard:
                self._active_run_id = None
                self.view = None

    def op_answer(self, op: dict[str, Any]) -> None:
        run_id = str(op.get("run_id") or "")
        answers = op.get("answers")
        if not isinstance(answers, list) or not answers:
            self._error("answer requires a non-empty answers list")
            return
        answers = [str(a) for a in answers]
        view = self.view
        if view is None or self._active_run_id != run_id:
            self._error(f"no pending question for run {run_id or '(missing)'}")
            return
        if not view.answer(answers):
            self._error(f"no pending question for run {run_id}")

    def op_cancel(self, op: dict[str, Any]) -> None:
        run_id = str(op.get("run_id") or "")
        if self._active_run_id != run_id:
            self._error(f"run {run_id or '(missing)'} is not active")
            return
        self.session.cancel()
        if self.view is not None:
            self.view.cancel_pending()

    # -- main loop --------------------------------------------------------------------

    def serve_forever(self) -> None:
        """Read ops until stdin EOF or a ``shutdown`` op.

        Writes the ``ready`` line first, then processes ops strictly in
        arrival order on this (reader) thread.  A ``submit`` spawns its
        own run thread and returns immediately, so the loop stays live
        for answer/cancel/ping/shutdown ops while the run executes.
        """
        self._write_line({"type": "ready", "pid": _pid()})
        while not self._stopped.is_set():
            line = self.inp.readline()
            if not line:  # EOF: the host closed the pipe
                break
            line = line.strip()
            if not line:
                continue
            try:
                op = json.loads(line)
            except ValueError:
                self._error("malformed op line")
                continue
            if not isinstance(op, dict) or not isinstance(op.get("op"), str):
                self._error("op must be an object with a string 'op' field")
                continue
            name = op["op"]
            if name == "submit":
                self.op_submit(op)
            elif name == "answer":
                self.op_answer(op)
            elif name == "cancel":
                self.op_cancel(op)
            elif name == "ping":
                self._write_line({"type": "pong"})
            elif name == "shutdown":
                self._stopped.set()
            else:
                self._error(f"unknown op: {name}")
        self._join_run_thread()

    def _join_run_thread(self, timeout: float = 10.0) -> None:
        """Wait briefly for the active run thread to unwind (shutdown)."""
        with self._active_guard:
            active = self._active_run_id
        if active is None:
            return
        self.session.cancel()
        if self.view is not None:
            self.view.cancel_pending()
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._active_guard:
                if self._active_run_id is None:
                    return
            time.sleep(0.05)

    def close(self) -> None:
        """Tear down (idempotent): flag stop and cancel any pending ask."""
        self._stopped.set()
        if self.view is not None:
            self.view.cancel_pending()


def _pid() -> int:
    import os

    return os.getpid()


def run_serve(
    session: Session,
    inp: TextIO | None = None,
    out: TextIO | None = None,
    err: TextIO | None = None,
    answer_timeout: float = DEFAULT_ANSWER_TIMEOUT,
) -> int:
    """Serve the resident protocol over *inp*/*out* until EOF/shutdown.

    Builds the ``AgentServer`` and runs the request loop on the calling
    thread.  ``answer_timeout`` bounds how long a pending ask/confirm
    waits for a host answer before falling back to "Unanswered"
    (0 = wait forever — the resident default, since a web user needs
    time to type; headless answers immediately instead).  Returns 0 on
    clean EOF/shutdown.
    """
    server = AgentServer(
        session,
        inp if inp is not None else sys.stdin,
        out if out is not None else sys.stdout,
        err=err,
        answer_timeout=answer_timeout,
    )
    try:
        server.serve_forever()
        return 0
    finally:
        server.close()
