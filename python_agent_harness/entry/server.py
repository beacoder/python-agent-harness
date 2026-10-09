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
  {"op": "answer", "run_id": ..., "answers": [...], "ask_id": ...}
  {"op": "cancel", "run_id": ...}
  {"op": "hello", "protocol_versions": [...]}
  {"op": "ping"} / {"op": "shutdown"}
Any op may carry an ``op_id``, echoed on the ``error`` (or ``pong``)
it causes so a refusal names which op it refused.
agent → host (one JSON object per line; ``protocol`` (the wire schema
version, see ``headless.PROTOCOL_VERSION``) stamps every line, ``seq``
on every run line):
  {"type": "ready", "pid": ..., "protocol_version": ...,
   "capabilities": [...]}                     first line, no run_id
  {"type": "start"|"delta"|"notify"|"log", "run_id": ...}
  {"type": "result", "run_id": ..., "answer": ...,
   "errors": [{"code", "message"}], "error_messages": [...], ...}
  {"type": "pong"|"error", ...}                control lines, no run_id;
      error carries {"error": {"code", "message"}, "message": ...}
  notify with kind "ask" carries {"kind": "ask"|"confirm", "ask_id", ...}
  notify with kind "usage" carries {"input", "output", "rounds"}

Capabilities: ``ready`` lists the named features this build supports
(see ``CAPABILITIES``).  ``protocol_version`` alone is a single integer
a host can only accept or reject wholesale; the list lets it discover
what is available and adapt, instead of inferring a feature's absence
from events that never arrive.

Asks are correlated: every ask/confirm line carries an ``ask_id``, and
``answer`` may echo it.  A mismatch is refused rather than applied to
whatever is pending now — without correlation a reply the host sent
for a question that has since timed out would resolve the NEXT one,
answering a question the user never saw.  Omitting ``ask_id`` keeps
the original behaviour, except once some ask in the run has timed out:
from then on an uncorrelated answer is genuinely ambiguous and is
refused with an ``error`` naming ``ask_id``.

Usage is incremental: a ``notify`` of kind ``usage`` carries the run's
running ``{input, output, rounds}`` after each round (sub-agent tokens
included).  The ``result`` line remains canonical for billing, but a
host can meter mid-run and ``cancel`` a run that outruns its budget
instead of only discovering the cost once it has finished.

Error codes are assigned where the outcome is known: the agent loop
notifies ``{"code", "message"}`` for classified failures (``budget``,
``timeout``), so a driver's branch does not depend on matching words
in a human-readable sentence.  Plain-string errors still fall back to
the text heuristic in ``headless._as_error``.

Version negotiation runs both ways.  ``ready`` announces what this
build speaks -- a single integer a host can only accept or reject
wholesale.  ``hello`` is the other direction: the host states the
versions IT can parse and the server confirms a shared one or refuses
with ``protocol``, so a mismatch is settled once, before any run,
rather than surfacing as misparsed events.  With one line shape today
the only outcomes are "agreed" or "no overlap", but it is the hook a
future version needs in order to downgrade instead of breaking an
older host.

There is no generic ``ack`` line: each op is acknowledged by its own
effect on the stream, so a host correlates by that rather than waiting
for a receipt.  Failures are the exception, because an effect that
never happened cannot be correlated: an op may carry an ``op_id``,
which is echoed on the ``error`` it causes.  A host that pipelined an
``answer`` and a ``cancel`` can then tell which one was refused
instead of seeing an unattributed error line.  ``submit`` is acknowledged by the run's ``start``
line (or an ``error`` if rejected); ``ping`` by ``pong``; ``answer``
and ``cancel`` only answer back on failure (an ``error`` line) — on
success they are observable through the run resuming or ending with
``cancelled: true``; ``shutdown`` by the stream ending (the final
``result`` of a drained run, then EOF).

Budgets: a run can be bounded by a round cap and a wall-clock limit
(``--max-rounds``/``--timeout``, or the config file's ``serve``
section; both off by default).  They apply per ``submit``, not per
process — the resident server grants each run a fresh budget.  They
are deliberately NOT fields on the submit op: ``serve`` sandboxes
untrusted agent code on behalf of a host, so the ceiling belongs to
whoever starts the sandbox, not to the caller.  A tripped budget
unwinds the agent loop through its normal path, so the run still ends
with a ``result`` line carrying ``errors[].code`` of ``budget`` or
``timeout`` plus the usage consumed.

The ``timeout`` is COOPERATIVE: it is checked between rounds, not by a
watchdog, so it bounds work at those checkpoints but does not forcibly
interrupt a blocking operation mid-round — a hung tool, a slow HTTP
read, or a run parked on a mid-run ``ask`` awaiting a human answer.
Bound those separately: ``--answer-timeout`` caps how long an ask
waits, and the hosting process's own exec watchdog (it already sends
``cancel`` then kills) is the hard wall-clock stop for anything the
cooperative budget cannot reach.

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

import contextlib
import json
import signal
import sys
import threading
import time
import uuid
from typing import Any, TextIO

from ..io.text_filter import strip_final_check
from ..session.session import Session
from .controller import Controller
from .headless import PROTOCOL_VERSION, JsonlView, _restorable_handler

DEFAULT_ANSWER_TIMEOUT = 0.0  # wait forever for a host answer

# Named features advertised on the ``ready`` line.  ``protocol_version``
# alone is a single integer a host can only accept or reject wholesale;
# these let it discover what this build actually supports and adapt,
# instead of inferring a feature's absence from events that never
# arrive.  Add a capability when behaviour a host can branch on is
# introduced; never repurpose an existing name.
CAPABILITIES = (
    "submit",  # run a prompt
    "answer",  # deliver a mid-run answer to an ask
    "cancel",  # protocol-level cancel (no signals)
    "ping",  # liveness probe, replies pong
    "shutdown",  # graceful drain, final result then EOF
    "ask_id",  # ask lines carry an id that `answer` may echo
    "usage_notify",  # per-round notify kind "usage" during a run
    "error_codes",  # result errors carry a structured {"code", "message"}
    "per_run_budget",  # sandbox-side max_rounds / timeout per submit
    "signal_drain",  # SIGINT/SIGTERM drain, emitting a final result
    "hello",  # host-side version negotiation (op:hello -> type:hello)
    "op_id",  # ops may carry an id, echoed on the error/pong they cause
)


class _ShutdownSignal(BaseException):
    """Raised in the reader thread by a SIGINT/SIGTERM handler.

    Derives from ``BaseException`` on purpose: it unwinds the reader
    loop and must not be swallowed by an ``except Exception`` guard on
    the way out.
    """


class graceful_signal_shutdown:
    """Context manager draining the server on SIGINT/SIGTERM.

    ``serve`` is resident, so a signal means "stop the server", not
    merely "cancel this run" — which is why headless's one-shot
    ``signal_canceller`` does not fit.  Two things have to happen, and
    a flag alone achieves neither: the reader loop is parked in a
    blocking ``readline`` (which Python retries after a handler
    returns, per PEP 475), so the handler *raises* to break out of it.
    ``serve_forever`` then reaches ``_join_run_thread``, which cancels
    the active run and waits for it to emit its terminal ``result``
    line — carrying ``cancelled: true`` and the run's real token
    usage.

    Without this, SIGTERM (``docker stop`` sends it to the sandbox's
    PID 1) killed the process mid-run through the default handler: no
    ``result`` line, so a driving host lost the partial answer, the
    salvaged history, and — because it bills off that line — every
    token the run had already consumed.

    One-shot: each handler uninstalls itself before raising, so a
    second signal takes its default action (immediate death) rather
    than interrupting the drain.  Only the main thread may install
    handlers; from any other thread (ValueError) or on a platform
    lacking the signal (OSError) this degrades to a no-op, matching
    ``signal_canceller``.
    """

    def __init__(self, server: AgentServer) -> None:
        self._server = server
        self._previous: dict[int, Any] = {}
        self._installed: list[int] = []

    def _handle(self, signum: int, frame: Any) -> None:
        # Uninstall first: a second signal must not land inside the
        # drain we are about to start.  Suppress broadly (TypeError
        # included) because the raise below is what the whole drain
        # depends on — a restore failure must never replace
        # _ShutdownSignal with an exception the loop does not expect,
        # which would crash serve_forever and skip the drain entirely.
        with contextlib.suppress(ValueError, OSError, TypeError):
            signal.signal(signum, self._previous.get(signum, signal.SIG_DFL))
        self._server._stopped.set()
        raise _ShutdownSignal

    def __enter__(self) -> graceful_signal_shutdown:
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(ValueError, OSError):
                self._previous[sig] = _restorable_handler(signal.signal(sig, self._handle))
                self._installed.append(sig)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for sig in reversed(self._installed):
            with contextlib.suppress(ValueError, OSError, TypeError):
                signal.signal(sig, self._previous.get(sig, signal.SIG_DFL))
        self._installed.clear()
        self._previous.clear()


class _AskState:
    """One pending interactive prompt awaiting a host answer.

    ``ServerView.confirm``/``ask`` (agent worker thread) create the
    state, emit the ask line, and block on ``wait``; the server's
    reader resolves it via ``ServerView.answer``.

    ``ask_id`` correlates the answer with the question.  Without it an
    ``answer`` could only target "whatever is pending", so a reply the
    host sent for an abandoned question would silently resolve the
    NEXT one — answering a question the user never saw.
    """

    def __init__(self, run_id: str | None, kind: str) -> None:
        self.run_id = run_id
        self.kind = kind  # "ask" | "confirm"
        self.ask_id = uuid.uuid4().hex[:12]
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
        # Latched when an ask times out: an answer with no ``ask_id``
        # becomes ambiguous from then on (see ``_abandon``).
        self._abandoned = False
        # Seconds to wait for a host answer before falling back to
        # "Unanswered" (headless semantics).  0 = wait forever.
        self.answer_timeout = answer_timeout
        # Set when the run is cancelled: a pending ask must unblock at
        # once (the agent loop is unwinding anyway).
        self._cancelled = threading.Event()

    # -- interactive prompts ---------------------------------------------------

    def _wait_answer(self, state: _AskState) -> str:
        """Block until answered, cancelled, or the deadline passes.

        Always retires *state* on the way out: once this returns, the
        question is no longer waiting for anything.  Leaving a resolved
        state in the pending slot would let a second answer land on a
        dead question, and would make ``pending_ask_id`` report an ask
        nobody is waiting on.
        """
        deadline = time.monotonic() + self.answer_timeout if self.answer_timeout > 0 else None
        try:
            while True:
                if state.event.wait(0.1):
                    # cancellation outranks a concurrent resolve (the run is
                    # unwinding; the agent must see an empty answer, TUI
                    # parity) — cancel may land while we are inside wait()
                    return "" if self._cancelled.is_set() else state.answer
                if self._cancelled.is_set():
                    return ""
                if deadline is not None and time.monotonic() >= deadline:
                    self._abandon(state)
                    return "Unanswered"
        finally:
            self._retire(state)

    def _retire(self, state: _AskState) -> None:
        """Drop *state* from the pending slot if it still owns it.

        Guarded on identity so a newer ask — published while this one
        was being resolved — is never discarded.
        """
        with self._pending_lock:
            if self._pending is state:
                self._pending = None

    def _abandon(self, state: _AskState) -> None:
        """Retire a timed-out ask so it can never be resolved later.

        Without this the timed-out state stays in the pending slot: a
        host answer arriving afterwards would be accepted, and once the
        agent published its NEXT question that stale reply would resolve
        it instead — delivering an answer to a question it was not
        written for.

        Also latches ``_abandoned``: from here on an answer that
        carries no ``ask_id`` is genuinely ambiguous (it could be meant
        for the question we just gave up on), so ``op_answer`` refuses
        it rather than guessing.  Correlation is the only real fix;
        this makes the uncorrelated case fail loudly instead of quietly
        answering the wrong question.
        """
        self._retire(state)
        with self._pending_lock:
            self._abandoned = True

    def requires_ask_id(self) -> bool:
        """Whether an answer must name its question to be accepted.

        True once any ask in this run has timed out — see ``_abandon``.
        Only reachable with a non-zero ``answer_timeout``; with the
        default (wait forever) no ask is ever abandoned, so an
        id-less answer stays acceptable.
        """
        with self._pending_lock:
            return self._abandoned

    def _safe_emit(self, payload: dict[str, Any]) -> None:
        """Emit a run line, tolerating a dead host pipe.

        ``_emit``/``_write`` raise on a closed stdout (host death or
        shutdown-while-running); a crashed worker callback would tear
        down the agent loop with a BrokenPipeError instead of ending
        the run cleanly.
        """
        with contextlib.suppress(BrokenPipeError, ValueError, OSError):
            self._emit(payload)

    def answer(self, answers: list[str], ask_id: str | None = None) -> bool:
        """Resolve the pending ask/confirm with *answers* (host-side).

        Routes by the pending state's kind: confirm takes the first
        answer verbatim (the caller decides yes/no), ask joins multiple
        values with ", " (mirroring the TUI's multi-select resolution).

        ``ask_id`` (when given) must name the question actually
        pending; a mismatch is refused rather than applied to whatever
        is waiting now.  Omitting it keeps the original "resolve
        whatever is pending" behaviour, so a host that predates the id
        still works.  Returns False when nothing is pending or the id
        does not match.
        """
        with self._pending_lock:
            state = self._pending
            if state is None:
                return False
            if ask_id is not None and ask_id != state.ask_id:
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

    def pending_ask_id(self) -> str | None:
        """The id of the question currently awaiting an answer, if any."""
        with self._pending_lock:
            return self._pending.ask_id if self._pending is not None else None

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
            self._sealed = False
        self._cancelled.clear()
        self.run_id = run_id
        self._deltas.reset()
        with self._pending_lock:
            self._abandoned = False

    # -- HeadlessView overrides -------------------------------------------------

    def confirm(self, prompt: str) -> bool:
        state = _AskState(self.run_id, "confirm")
        with self._pending_lock:
            self._pending = state
        self._safe_emit(
            {
                "type": "notify",
                "kind": "ask",
                "data": {"kind": "confirm", "prompt": prompt, "ask_id": state.ask_id},
            }
        )
        answer = self._wait_answer(state)
        return answer.strip().lower() in ("y", "yes", "true", "1")

    def ask(self, questions: list[dict]) -> str:
        state = _AskState(self.run_id, "ask")
        with self._pending_lock:
            self._pending = state
        self._safe_emit(
            {
                "type": "notify",
                "kind": "ask",
                "data": {"kind": "ask", "questions": questions, "ask_id": state.ask_id},
            }
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
        max_rounds: int | None = None,
        timeout: float | None = None,
    ) -> None:
        self.session = session
        self.controller = Controller(session)
        self.inp = inp
        self.out = out
        self.err = err if err is not None else sys.stderr
        self.answer_timeout = answer_timeout
        # Per-run budgets (None = unlimited), applied to EVERY submit.
        # The process is resident, so these bound each run rather than
        # the process lifetime.  Held here — not taken off the wire —
        # so a driving host cannot raise its own ceiling.
        self.max_rounds = max_rounds
        self.timeout = timeout
        self.view: ServerView | None = None
        self._active_run_id: str | None = None
        self._active_guard = threading.Lock()  # active-run transitions
        # A cancel op that arrived before the run thread reached
        # ``Controller.submit``: submit() clears the session cancel
        # event at run start, which would swallow the cancel (verified
        # race).  The run thread re-applies it right after submit.
        self._cancel_pending = threading.Event()
        self._stopped = threading.Event()
        # Versions the HOST said it can parse (op:hello).  None until it
        # greets us; informational today, since there is one line shape,
        # but it is what a future version would downgrade against.
        self.host_protocol_versions: list[int] | None = None

    # -- outbound lines ---------------------------------------------------------

    def _write_line(self, payload: dict[str, Any]) -> None:
        """Write one protocol line, serialized against run-thread output.

        Live events (start/delta/notify/log/result) are written by the
        run's threads under the attached view's ``_lock``; control
        lines (ready/pong/error) come from the reader thread.  Taking
        the same lock here keeps every line whole — without it an
        ``error`` emitted mid-run could interleave with a streamed
        event line on the wire.

        A dead host (EOF/closed pipe) makes writes raise; that is not
        a server fault — the pump sees EOF and the loop exits.  Swallow
        the write error so unwinding threads (result line after host
        death, error inside an except handler) never die with a
        secondary BrokenPipeError traceback.

        Control lines carry ``protocol`` (the wire schema version);
        live run lines get ``seq``/``run_id`` via the view's ``_write``.
        """
        text = (
            json.dumps({"protocol": PROTOCOL_VERSION, **payload}, ensure_ascii=False, default=str)
            + "\n"
        )
        view = self.view
        try:
            if view is not None:
                with view._lock:
                    self.out.write(text)
                    self.out.flush()
            else:
                self.out.write(text)
                self.out.flush()
        except (BrokenPipeError, ValueError, OSError):
            pass  # host is gone; the reader loop's EOF handles shutdown

    def _error(self, code: str, message: str, op_id: str | None = None) -> None:
        """Emit a protocol ``error`` line, echoing *op_id* when given.

        There is deliberately no generic ``ack`` on this protocol: each
        op is acknowledged by its own effect on the stream.  That leaves
        failures unattributable, though -- a host that pipelined an
        ``answer`` and a ``cancel`` sees a bare ``error`` and cannot
        tell which one was refused, and the web driver folds it into the
        run's error trail beside genuine agent errors.  Echoing the
        caller's own ``op_id`` fixes attribution without reintroducing
        receipts for the success path: still no ack, but a refusal now
        names what it refused.
        """
        payload: dict[str, Any] = {
            "type": "error",
            "error": {"code": code, "message": message},
            "message": message,
        }
        if op_id is not None:
            payload["op_id"] = op_id
        self._write_line(payload)

    @staticmethod
    def _op_id(op: dict[str, Any]) -> str | None:
        """The caller's correlation id for this op, if it supplied one."""
        value = op.get("op_id")
        return str(value) if value is not None else None

    # -- run execution ------------------------------------------------------------

    def _cancelled_now(self) -> bool:
        event = getattr(self.session, "cancel_event", None)
        return event is not None and event.is_set() is True

    def _final_answer(self) -> str:
        """The run's final assistant answer, TUI-filtered (headless shape).

        Skips assistant messages that strip to empty (e.g. a trailing
        check-only [FINAL CHECK] message) in favor of the real answer
        before them — see ``headless.final_answer_text``.
        """
        for msg in reversed(self.session.last_messages):
            if getattr(msg, "role", None) != "assistant":
                continue
            text = strip_final_check(msg.text_without_reasoning()).strip()
            if text:
                return text
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
        # Per-run budget: Controller.submit derives budget_top_level
        # from max_rounds being set, so passing it is what opts this
        # run into a finite round cap (default None = unlimited, the
        # interactive behavior).  A tripped budget unwinds the agent
        # loop normally, so the run still ends with a result line
        # carrying errors[].code = "budget"/"timeout" and real usage.
        handle = self.controller.submit(
            prompt,
            max_rounds=self.max_rounds,
            timeout=self.timeout,
        )
        if self._cancel_pending.is_set():
            # A cancel op landed before submit() — submit cleared the
            # session cancel event at run start, so re-apply it now
            # (the run must unwind; TUI parity for a Ctrl-C at turn start).
            self._cancel_pending.clear()
            self.session.cancel()
        model = self.session.model
        if handle is None:
            # Failure before start, headless contract: the result line is
            # the only line of the run (seq 1, run_id echoed).  Nothing
            # ran, so usage is zeros — the previous run's totals must
            # not leak onto this run's result line.
            view.emit_result(
                "",
                errors=["nothing to send"],
                usage={"input": 0, "output": 0, "rounds": 0},
                model=model,
            )
            return
        # Flush events the worker emitted between submit() and now (the
        # emit_start contract: start is guaranteed to be the first line).
        view.emit_start(prompt, list(handle.warnings))
        handle.worker.join()
        # Snapshot usage AFTER the run: Controller.submit() swaps in a
        # fresh zeroed totals dict, so a pre-run snapshot would always
        # report zeros (billing data lost).  The run mutates the dict
        # in place; read the CURRENT values here.
        usage = self._usage_snapshot()
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
        op_id = self._op_id(op)
        run_id = str(op.get("run_id") or "")
        if not run_id:
            self._error("protocol", "submit requires run_id", op_id)
            return
        prompt = op.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            self._error("protocol", "submit requires a non-empty prompt", op_id)
            return
        with self._active_guard:
            if self._active_run_id is not None:
                self._error("protocol", "a run is already active", op_id)
                return
            self._active_run_id = run_id
        # daemon: the run waits on the agent worker, which can block in
        # an uninterruptible tool (a stuck syscall ignores the cancel
        # event).  _join_run_thread already bounds how long shutdown
        # waits for it; without daemon the thread would ALSO keep the
        # interpreter alive past that budget at exit, so `shutdown`
        # could never terminate the process and the host would have to
        # escalate to SIGKILL.
        thread = threading.Thread(
            target=self._run_thread,
            args=(prompt, run_id),
            name=f"serve-run-{run_id}",
            daemon=True,
        )
        thread.start()

    def _run_thread(self, prompt: str, run_id: str) -> None:
        try:
            self._execute_run(prompt, run_id)
        except Exception as e:  # noqa: BLE001 - a run failure must not kill the server
            self._error("protocol", f"run failed: {e}")
        finally:
            # Unwire the session callbacks before dropping the view: a
            # worker still unwinding from this run (a detached
            # sub-agent thread, a tool finishing after a cancel) would
            # otherwise emit through session.notify_fn — which the NEXT
            # run's attach_view repoints at ITS view, so run N's
            # straggler would land in run N+1's stream wearing run
            # N+1's run_id/seq.
            #
            # ORDER MATTERS: the detach must happen BEFORE
            # _active_run_id is cleared.  op_submit refuses a new run
            # while that id is set, so detaching first guarantees we
            # unwire THIS run's callbacks and never the next run's.
            # Swapping these two would reintroduce the contamination
            # it is here to prevent.  Suppressed broadly because this
            # is a finally: a raise here would leave _active_run_id
            # set and wedge the server ("a run is already active"
            # forever).
            with contextlib.suppress(Exception):
                self.controller.detach_view()
            with self._active_guard:
                self._active_run_id = None
                self.view = None
            self._cancel_pending.clear()

    def op_answer(self, op: dict[str, Any]) -> None:
        op_id = self._op_id(op)
        run_id = str(op.get("run_id") or "")
        answers = op.get("answers")
        if not isinstance(answers, list) or not answers:
            self._error("protocol", "answer requires a non-empty answers list", op_id)
            return
        answers = [str(a) for a in answers]
        ask_id = op.get("ask_id")
        ask_id = str(ask_id) if ask_id is not None else None
        view = self.view
        if view is None or self._active_run_id != run_id:
            self._error("protocol", f"no pending question for run {run_id or '(missing)'}", op_id)
            return
        if ask_id is None and view.requires_ask_id():
            # An earlier ask in this run timed out, so we cannot tell
            # which question this reply is for.  Refuse instead of
            # applying it to whatever happens to be waiting now.
            self._error(
                "protocol",
                "answer is ambiguous: a previous question timed out in this "
                "run, so answers must carry the ask_id from the ask line",
                op_id,
            )
            return
        if not view.answer(answers, ask_id=ask_id):
            # Distinguish a stale answer from no question at all: the
            # host sent a reply for a question that is no longer the one
            # waiting (it timed out, or was already answered), and
            # applying it to the current question would answer the
            # wrong thing.
            pending = view.pending_ask_id()
            if ask_id is not None and pending is not None and pending != ask_id:
                self._error(
                    "protocol",
                    f"stale answer: ask_id {ask_id} is not the pending "
                    f"question ({pending}) for run {run_id}",
                    op_id,
                )
            else:
                self._error("protocol", f"no pending question for run {run_id}", op_id)

    def op_cancel(self, op: dict[str, Any]) -> None:
        op_id = self._op_id(op)
        run_id = str(op.get("run_id") or "")
        if self._active_run_id != run_id:
            self._error("protocol", f"run {run_id or '(missing)'} is not active", op_id)
            return
        # The run thread may not have reached Controller.submit yet (it
        # clears the cancel event at run start); remember the intent so
        # _execute_run re-applies it after submit.
        self._cancel_pending.set()
        self.session.cancel()
        if self.view is not None:
            self.view.cancel_pending()

    # -- main loop --------------------------------------------------------------------

    def op_hello(self, op: dict[str, Any]) -> None:
        """Negotiate the wire version with the host.

        ``ready`` announces what this build speaks, which a host can
        only accept or reject wholesale.  ``hello`` is the other
        direction: the host states the versions IT can parse, and this
        server either confirms one it shares or refuses -- so a
        mismatch is settled once, before any run, instead of surfacing
        as misparsed events.

        It is a no-op for a single-version world (there is one shape
        today, so the only outcomes are "agreed on 1" or "no overlap"),
        but it is the hook a future version needs to downgrade its line
        shapes instead of breaking an older host.
        """
        op_id = self._op_id(op)
        raw = op.get("protocol_versions")
        if raw is None:
            raw = op.get("protocol_version")
        candidates: list[Any] = (
            list(raw) if isinstance(raw, list) else ([] if raw is None else [raw])
        )
        # Narrow before converting rather than letting ``int()`` raise on
        # whatever arrives: the op is untrusted input, so the accepted
        # shapes belong in the code instead of in an ``except TypeError``.
        # An uncaught conversion error here would unwind the reader loop
        # and take the whole sandbox down with it.
        versions: list[int] = []
        for value in candidates:
            if isinstance(value, bool):
                continue  # False would otherwise be read as version 0
            if isinstance(value, int):
                versions.append(value)
            elif isinstance(value, float):
                # is_integer() is False for inf and nan too, which
                # ``json`` accepts by default and ``int()`` refuses
                if value.is_integer():
                    versions.append(int(value))
            elif isinstance(value, str):
                with contextlib.suppress(ValueError):
                    versions.append(int(value.strip()))
        if not versions:
            # A hello with no parseable version is a greeting, not a
            # negotiation: confirm what we speak and carry on.
            versions = [PROTOCOL_VERSION]
        if PROTOCOL_VERSION not in versions:
            self.host_protocol_versions = versions
            self._error(
                "protocol",
                f"no shared protocol version: host accepts {sorted(versions)}, "
                f"this build speaks {PROTOCOL_VERSION}",
                op_id,
            )
            return
        self.host_protocol_versions = versions
        payload: dict[str, Any] = {
            "type": "hello",
            "protocol_version": PROTOCOL_VERSION,
            "capabilities": list(CAPABILITIES),
            "pid": _pid(),
        }
        if op_id is not None:
            payload["op_id"] = op_id
        self._write_line(payload)

    def serve_forever(self) -> None:
        """Read ops until stdin EOF, a ``shutdown`` op, or a signal.

        Writes the ``ready`` line first (``protocol`` on every control
        line, ``pid`` on ready), then processes ops strictly in
        arrival order on this (reader) thread.  A ``submit`` spawns its
        own run thread and returns immediately, so the loop stays live
        for answer/cancel/ping/shutdown ops while the run executes.

        SIGINT/SIGTERM drain rather than kill: the handler breaks the
        blocking ``readline`` so ``_join_run_thread`` can cancel the
        active run and let it emit its terminal ``result`` line (with
        ``cancelled: true`` and the run's token usage) before the
        process exits.  See ``graceful_signal_shutdown``.
        """
        self._write_line(
            {
                "type": "ready",
                "pid": _pid(),
                "protocol_version": PROTOCOL_VERSION,
                "capabilities": list(CAPABILITIES),
            }
        )
        with graceful_signal_shutdown(self), contextlib.suppress(_ShutdownSignal):
            # _ShutdownSignal: a signal asked us to stop mid-readline;
            # fall through to the drain below so the active run still
            # emits its result line.
            self._read_ops()
        self._join_run_thread()

    def _read_ops(self) -> None:
        """The reader loop proper: dispatch ops until EOF/shutdown."""
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
                self._error("protocol", "malformed op line")
                continue
            if not isinstance(op, dict) or not isinstance(op.get("op"), str):
                op_id = self._op_id(op) if isinstance(op, dict) else None
                self._error("protocol", "op must be an object with a string 'op' field", op_id)
                continue
            name = op["op"]
            if name == "submit":
                self.op_submit(op)
            elif name == "answer":
                self.op_answer(op)
            elif name == "cancel":
                self.op_cancel(op)
            elif name == "hello":
                self.op_hello(op)
            elif name == "ping":
                pong: dict[str, Any] = {"type": "pong"}
                op_id = self._op_id(op)
                if op_id is not None:
                    pong["op_id"] = op_id
                self._write_line(pong)
            elif name == "shutdown":
                self._stopped.set()
            else:
                self._error("protocol", f"unknown op: {name}", self._op_id(op))

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
    max_rounds: int | None = None,
    timeout: float | None = None,
) -> int:
    """Serve the resident protocol over *inp*/*out* until EOF/shutdown.

    Builds the ``AgentServer`` and runs the request loop on the calling
    thread.  ``answer_timeout`` bounds how long a pending ask/confirm
    waits for a host answer before falling back to "Unanswered"
    (0 = wait forever — the resident default, since a web user needs
    time to type; headless answers immediately instead).

    ``max_rounds``/``timeout`` opt EACH run into a round budget and a
    wall-clock limit (both default off, matching headless and the
    interactive TUI).  They are per-submit, not per-process: the
    resident process serves many runs and each gets a fresh budget.
    Enforced here rather than read off the wire so a driving host
    cannot raise its own ceiling.  Returns 0 on clean EOF/shutdown.
    """
    server = AgentServer(
        session,
        inp if inp is not None else sys.stdin,
        out if out is not None else sys.stdout,
        err=err,
        answer_timeout=answer_timeout,
        max_rounds=max_rounds,
        timeout=timeout,
    )
    try:
        server.serve_forever()
        return 0
    finally:
        server.close()
