"""Headless (non-interactive) view and runner for the agent harness.

A minimal ``View`` implementation for CI / scripting / piping: the
filtered final assistant answer is written to stdout (the reasoning
preamble and the trailing [FINAL CHECK] block are stripped, mirroring
the TUI's display filters); tool/status events go to stderr.
Interactive prompts are auto-answered (``confirm`` -> True, ``ask`` ->
"Unanswered") so a run never blocks waiting for a human.

The runner drives one prompt through ``Controller.submit`` and waits
for the worker thread, mirroring the TUI's run lifecycle without any
of the rendering.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from typing import Any, TextIO

from .controller import Controller
from .session import Session
from .text_filter import strip_final_check


def final_answer_text(session: Session) -> str:
    """The run's final assistant answer, TUI-filtered.

    Takes the last assistant message from the session history, strips
    the reasoning preamble (``text_without_reasoning``) and the
    trailing [FINAL CHECK] block, and returns the result.  Empty when
    the run produced no assistant message (e.g. it errored out).
    """
    for msg in reversed(session.last_messages):
        if getattr(msg, "role", None) != "assistant":
            continue
        return strip_final_check(msg.text_without_reasoning()).strip()
    return ""


class HeadlessView:
    """A ``View`` that renders to plain text streams.

    Assistant deltas are ignored (the filtered final answer is written
    to *out* by ``run_headless`` once the run completes); tool/status
    notifications and log lines go to *err* (stderr by default) so the
    answer stream stays clean for piping.  ``confirm`` auto-approves
    and ``ask`` returns "Unanswered" — headless runs never block on
    input.
    """

    def __init__(self, out: TextIO | None = None, err: TextIO | None = None) -> None:
        self.out = out if out is not None else sys.stdout
        self.err = err if err is not None else sys.stderr
        # "error" notifications seen during the run; run_headless uses
        # this to return a non-zero exit code for CI.
        self.errors: list[str] = []

    def on_delta(self, text: str) -> None:
        pass

    def on_notify(self, kind: str, data: Any = None) -> None:
        if kind == "tool_start":
            names = data if isinstance(data, list) else []
            label = ", ".join(names) if names else "tools"
            self.err.write(f"\n[tools: {label}]\n")
        elif kind == "error":
            self.errors.append(str(data))
            self.err.write(f"\n[error: {data}]\n")
        elif kind == "run_done":
            self.err.write("\n[done]\n")
        self.err.flush()

    def on_log(self, msg: str) -> None:
        self.err.write(f"\n[log: {msg}]\n")
        self.err.flush()

    def confirm(self, prompt: str) -> bool:
        return True

    def ask(self, questions: list[dict]) -> str:
        return "Unanswered"

    def run(self) -> None:
        pass


def restore_session(controller: Controller, spec: str, err: TextIO) -> bool:
    """Load a saved session into *controller* so a run continues it.

    SPEC is a file path, a title substring, or empty/"latest" for the
    most recently saved session.  Mirrors the TUI's /restore: rebuilds
    the conversation history, repoints the session store at the saved
    file, and restores the saved agent/model.  Returns False (with a
    message on *err*) when no matching session exists.
    """
    from .persistence import (
        SessionPersistence,
        find_session_by_title,
        parse_saved_body,
        title_from_filename,
    )

    if not spec or spec in ("--latest", "latest"):
        path = SessionPersistence.latest_session()
    elif os.path.isfile(os.path.expanduser(spec)):
        path = os.path.expanduser(spec)
    else:
        path = find_session_by_title(spec)
    if not path or not os.path.isfile(path):
        err.write(f"no session found: {spec or 'latest'}\n")
        return False
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    meta = SessionPersistence.parse_metadata(text)
    body = SessionPersistence.strip_metadata(text)
    messages = parse_saved_body(body)
    controller.store.file_path = path
    title = title_from_filename(path)
    if title:
        controller.store.title = title
    agent = meta.get("python-agent-harness--agent")
    if agent:
        success, msg = controller.switch_agent(agent)
        if not success:
            controller.switch_agent("default")
            err.write(f"warning: {msg}\n")
    model = meta.get("python-agent-harness--model")
    if model and model != controller.model:
        if model == controller.llm_settings.get("model"):
            # The saved model is the session-start default: the
            # "default" pseudo-profile restores exactly those settings.
            controller.switch_model("default")
        else:
            success, _ = controller.switch_model(model)
            if not success:
                profile = next(
                    (
                        name
                        for name, p in controller.model_profiles.items()
                        if p.get("model") == model
                    ),
                    None,
                )
                if profile:
                    success, _ = controller.switch_model(profile)
            if not success:
                err.write(
                    f"warning: model {model!r} has no matching profile, keeping current model\n"
                )
    # Replace conversation history: a new generation, so a stale worker
    # from a previous run can't clobber the restored session.
    controller.run_generation += 1
    controller.last_messages = messages
    controller.clear_todos()
    err.write(f"restored: {os.path.basename(path)} ({len(messages)} messages)\n")
    return True


def _select_model(controller: Controller, name: str, err: TextIO) -> None:
    """Select the model for a headless run.

    NAME is first tried as a profile from the ``models`` config section
    (same semantics as the TUI's /model).  When no such profile exists,
    it is treated as a raw model name: the client's model is swapped
    while keeping the current base_url/api_key (a note is written to
    *err*).  Always succeeds.
    """
    success, _ = controller.switch_model(name)
    if success:
        return
    # No such profile: treat NAME as a raw model name on the current
    # endpoint (keeps base_url/api_key/temperature/etc. as-is).
    session = controller.session
    session.client.model = name
    session.model = name
    session.store.model = name
    session.calibrator.reset()
    err.write(f"note: no profile {name!r}, using model name directly\n")


def run_headless(
    session: Session,
    prompt: str,
    out: TextIO | None = None,
    err: TextIO | None = None,
    restore: str | None = None,
    model: str | None = None,
) -> int:
    """Run one prompt headlessly; return the process exit code.

    Submits *prompt* through a ``Controller`` wired to a
    ``HeadlessView``, waits for the run to finish, and writes the
    filtered final assistant answer to *out*.  Returns 0 on success.
    When *restore* is given, a saved session is loaded first so the
    run continues that conversation.  When *model* is given, that
    profile (or raw model name) is selected after the restore, so an
    explicit choice wins over the restored session's model.  Returns
    1 when the prompt was only failed ``@file`` references (nothing
    to send), the run raised an agent error, or the restore failed.
    """
    controller = Controller(session)
    view = HeadlessView(out=out, err=err)
    controller.attach_view(view)
    if restore is not None and not restore_session(controller, restore, view.err):
        return 1
    if model is not None:
        _select_model(controller, model, view.err)
    handle = controller.submit(prompt)
    if handle is None:
        view.err.write("nothing to send\n")
        view.err.flush()
        return 1
    for warning in handle.warnings:
        view.err.write(f"warning: {warning}\n")
    view.err.flush()
    handle.worker.join()
    answer = final_answer_text(session)
    if answer:
        view.out.write(answer + "\n")
        view.out.flush()
    return 1 if view.errors else 0


class JsonlView(HeadlessView):
    """A ``View`` that emits the run as JSON lines on *out*.

    ``--json`` mode for ``headless``: every event — streamed deltas,
    tool/status notifications, log lines, submit warnings, and the
    final result — becomes one ``{"type": ...}`` JSON object per line
    on stdout, so a driving process (CI, an IDE, a web backend) gets a
    structured event stream on one pipe.  Human-readable diagnostics
    (restore notes, model-switch notes) still go to *err* as plain
    text, and stderr also keeps the plain-text echo of error events so
    a failed run is diagnosable without JSON parsing.  ``confirm``
    auto-approves and ``ask`` returns "Unanswered", like
    ``HeadlessView``.

    Line kinds: ``start`` (echoes the prompt and submit warnings; the
    first line whenever a run actually starts), ``delta``, ``notify``
    (with ``kind``/``data`` as emitted by the session — ``data`` may be
    any JSON-serializable value or None), ``log``, and a final
    ``result`` (with the filtered answer and ``errors``).  The stream
    ends after ``result`` — which is also the only line when the run
    fails before start (restore failure, nothing to send).

    Events that arrive before ``emit_start`` (the worker thread starts
    as soon as the prompt is submitted) are buffered and flushed right
    behind the ``start`` line, so ``start`` is guaranteed to be first.
    A lock serializes buffer access and stream writes: the agent's
    worker thread emits live events while the caller's thread runs
    ``emit_start``/``emit_result``.
    """

    def __init__(self, out: TextIO | None = None, err: TextIO | None = None) -> None:
        super().__init__(out=out, err=err)
        self._lock = threading.Lock()
        # None = live mode (write through); a list = buffering mode.
        self._buffer: list[dict[str, Any]] | None = []

    def _write(self, payload: dict[str, Any]) -> None:
        self.out.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        self.out.flush()

    def _emit(self, payload: dict[str, Any]) -> None:
        with self._lock:
            if self._buffer is None:
                self._write(payload)
            else:
                self._buffer.append(payload)

    # deltas carry no meaning beyond concatenation, but stream them so
    # a driver can render progress; the result line remains canonical.
    def on_delta(self, text: str) -> None:
        self._emit({"type": "delta", "text": text})

    def on_notify(self, kind: str, data: Any = None) -> None:
        self._emit({"type": "notify", "kind": kind, "data": data})
        # Mirror "error" onto stderr as plain text and record it (the
        # parent's contract) so a failed run is diagnosable without
        # parsing JSON and the exit code still signals the failure.
        if kind == "error":
            with self._lock:
                self.errors.append(str(data))
            self.err.write(f"\n[error: {data}]\n")
            self.err.flush()

    def on_log(self, msg: str) -> None:
        self._emit({"type": "log", "message": msg})

    def emit_start(self, prompt: str, warnings: list[str]) -> None:
        """Write the ``start`` line, then any events buffered before it."""
        with self._lock:
            self._write({"type": "start", "prompt": prompt, "warnings": warnings})
            buffered, self._buffer = self._buffer, None
            for payload in buffered or []:
                self._write(payload)

    def emit_result(self, answer: str, errors: list[str] | None = None) -> None:
        """Write the terminal ``result`` line (always written, even when
        the run failed before ``emit_start``)."""
        with self._lock:
            self._buffer = None
            self._write(
                {
                    "type": "result",
                    "answer": answer,
                    "errors": list(self.errors if errors is None else errors),
                }
            )

    def run(self) -> None:
        pass


def run_headless_jsonl(
    session: Session,
    prompt: str,
    out: TextIO | None = None,
    err: TextIO | None = None,
    restore: str | None = None,
    model: str | None = None,
) -> int:
    """Run one prompt headlessly with a JSON-lines event stream on *out*.

    The ``--json`` counterpart of ``run_headless``: same submit/restore/
    model-selection flow and exit-code semantics, but ``JsonlView``
    replaces plain-text output — see its docstring for the line kinds.
    Exit code is 0 unless the prompt had nothing to send, the restore
    failed, or an ``error`` notification occurred (same rules as
    ``run_headless``).
    """
    controller = Controller(session)
    view = JsonlView(out=out, err=err)
    controller.attach_view(view)
    if restore is not None and not restore_session(controller, restore, view.err):
        view.emit_result("", errors=["restore failed"])
        return 1
    if model is not None:
        _select_model(controller, model, view.err)
    handle = controller.submit(prompt)
    if handle is None:
        view.emit_result("", errors=["nothing to send"])
        return 1
    view.emit_start(prompt, list(handle.warnings))
    handle.worker.join()
    view.emit_result(final_answer_text(session))
    return 1 if view.errors else 0
