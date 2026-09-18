"""Headless (non-interactive) view and runner for the agent harness.

A minimal ``View`` implementation for CI / scripting / piping: streams
assistant text to stdout, tool/status events to stderr, and auto-answers
interactive prompts (``confirm`` -> True, ``ask`` -> "Unanswered") so a
run never blocks waiting for a human.

The runner drives one prompt through ``Controller.submit`` and waits for
the worker thread, mirroring the TUI's run lifecycle without any of the
rendering.
"""

from __future__ import annotations

import os
import sys
from typing import Any, TextIO

from .controller import Controller
from .session import Session


class HeadlessView:
    """A ``View`` that renders to plain text streams.

    Assistant deltas go to *out* (stdout by default); tool/status
    notifications and log lines go to *err* (stderr by default) so the
    answer stream stays clean for piping.  ``confirm`` auto-approves and
    ``ask`` returns "Unanswered" — headless runs never block on input.
    """

    def __init__(self, out: TextIO | None = None, err: TextIO | None = None) -> None:
        self.out = out if out is not None else sys.stdout
        self.err = err if err is not None else sys.stderr
        # "error" notifications seen during the run; run_headless uses
        # this to return a non-zero exit code for CI.
        self.errors: list[str] = []

    def on_delta(self, text: str) -> None:
        self.out.write(text)
        self.out.flush()

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


def run_headless(
    session: Session,
    prompt: str,
    out: TextIO | None = None,
    err: TextIO | None = None,
    restore: str | None = None,
) -> int:
    """Run one prompt headlessly; return the process exit code.

    Submits *prompt* through a ``Controller`` wired to a
    ``HeadlessView``, waits for the run to finish, and returns 0 on
    success.  When *restore* is given, a saved session is loaded first
    so the run continues that conversation.  Returns 1 when the prompt
    was only failed ``@file`` references (nothing to send), the run
    raised an agent error, or the restore failed.
    """
    controller = Controller(session)
    view = HeadlessView(out=out, err=err)
    controller.attach_view(view)
    if restore is not None and not restore_session(controller, restore, view.err):
        return 1
    handle = controller.submit(prompt)
    if handle is None:
        view.err.write("nothing to send\n")
        view.err.flush()
        return 1
    for warning in handle.warnings:
        view.err.write(f"warning: {warning}\n")
    view.err.flush()
    handle.worker.join()
    return 1 if view.errors else 0
