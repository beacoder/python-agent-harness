"""Application controller: the mediator between the model (``Session``)
and a ``View``.

``Controller`` owns the ``Session`` and exposes the state reads and
actions the presentation layer needs.  A view (the Rich TUI today, a
headless/CI view later) talks only to the controller — never to the
session directly — so the concrete view is swappable without touching
the model or the controller.

The controller also owns the run lifecycle: ``submit()`` parses
``@file`` references, builds the user message, bumps the run
generation, and spawns the worker thread that drives the agent loop.
A view calls ``submit(text)`` and then drives its own display loop on
the returned ``RunHandle`` (the worker thread).  This is what makes a
headless/CI view possible: it can submit runs and collect output
without the Rich TUI.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from ..core.agent import run_agent_loop
from ..core.models import ImagePart, Message, TextPart
from ..io.attachments import AttachmentError, parse_at_references
from ..session.session import Session
from .view import View


@dataclass
class RunHandle:
    """Handle to a running agent run, returned by ``Controller.submit``.

    ``worker`` is the daemon thread driving the agent loop; the view
    waits on it (``worker.is_alive()`` / ``worker.join()``) to know
    when the run finishes.  ``seq`` is the run's generation number
    (matches ``Controller.run_generation`` at submit time); a view can
    use it to detect staleness.  ``display_text`` is the user-facing
    text for the round (without image data).  ``errors`` are the
    ``@file`` validation failures the view should surface; ``warnings``
    are non-fatal notices (e.g. the active model cannot see the
    attached images).
    """

    worker: threading.Thread
    seq: int
    display_text: str
    errors: list[AttachmentError]
    warnings: list[str]


class Controller:
    """Mediator between a ``Session`` (model) and a ``View``.

    Owns the session, wires its callbacks to the attached view, and
    exposes read-only state accessors plus the actions a view can
    trigger (mode/model/agent switching, compaction, cancellation, ...).
    """

    def __init__(self, session: Session) -> None:
        self.session = session
        # The conversation as the agent loop sees it: the user message
        # of each round plus everything the loop produced.  Synced back
        # from ``session.last_messages`` after each run (which may have
        # compacted/salvaged it).  Owned here — not by a view — so any
        # view (TUI, headless) drives runs from the same history.
        self.conversation_history: list[Message] = []

    # ------------------------------------------------------------------
    # view wiring
    # ------------------------------------------------------------------
    def attach_view(self, view: View) -> None:
        """Point the session's callbacks at *view*'s methods.

        The session invokes these from its worker thread; the view is
        responsible for any thread-safety its rendering requires.
        """
        self.session.on_delta = view.on_delta
        self.session.notify_fn = view.on_notify
        self.session.log_fn = view.on_log
        self.session.confirm_fn = view.confirm
        self.session.ask_fn = view.ask

    # ------------------------------------------------------------------
    # read-only state accessors
    # ------------------------------------------------------------------
    @property
    def model(self) -> str:
        return self.session.model

    @property
    def model_profiles(self) -> dict[str, dict]:
        return self.session.model_profiles

    @model_profiles.setter
    def model_profiles(self, value: dict[str, dict]) -> None:
        self.session.model_profiles = value

    @property
    def llm_settings(self) -> dict:
        return self.session.llm_settings

    @property
    def client(self) -> Any:
        return self.session.client

    @property
    def project_dir(self) -> str:
        return self.session.project_dir

    @project_dir.setter
    def project_dir(self, value: str) -> None:
        self.session.project_dir = value

    @property
    def last_messages(self) -> list:
        return self.session.last_messages

    @last_messages.setter
    def last_messages(self, value: list) -> None:
        self.session.last_messages = value

    @property
    def todos(self) -> list[dict]:
        return self.session.todos

    @property
    def context_ratio(self) -> float | None:
        return self.session.context_ratio

    @property
    def plan_mode(self) -> Any:
        return self.session.plan_mode

    @property
    def store(self) -> Any:
        return self.session.store

    @property
    def system_prompt(self) -> str | None:
        return self.session.system_prompt

    @property
    def supports_image_input(self) -> bool:
        return self.session.supports_image_input

    @property
    def config_path(self) -> str | None:
        return self.session.config_path

    @property
    def configured_context_path(self) -> str | None:
        return self.session._configured_context_path

    @property
    def startup_warnings(self) -> list[str]:
        return self.session.startup_warnings

    @property
    def cancel_event(self) -> Any:
        return self.session.cancel_event

    @property
    def run_generation(self) -> int:
        return self.session.run_generation

    @run_generation.setter
    def run_generation(self, value: int) -> None:
        self.session.run_generation = value

    @property
    def save_error(self) -> str | None:
        return self.session._save_error

    # ------------------------------------------------------------------
    # actions
    # ------------------------------------------------------------------
    def switch_to_plan(self) -> None:
        self.session.switch_to_plan()

    def switch_to_build(self) -> None:
        self.session.switch_to_build()

    def switch_model(self, name: str) -> tuple[bool, str]:
        return self.session.switch_model(name)

    def switch_agent(self, name: str) -> tuple[bool, str]:
        return self.session.switch_agent(name)

    def compact_conversation(self) -> tuple[bool, str]:
        return self.session.compact_conversation()

    def summarize_conversation(self) -> str:
        return self.session.summarize_conversation()

    def cancel(self) -> None:
        self.session.cancel()

    def clear_todos(self) -> None:
        self.session.clear_todos()

    def save(self, text: str) -> str | None:
        return self.session.store.save(text)

    def clear_conversation(self) -> None:
        """Reset the conversation to empty (a new run generation)."""
        self.session.run_generation += 1
        self.session.last_messages = []
        self.conversation_history = []
        self.session.clear_todos()

    # ------------------------------------------------------------------
    # run lifecycle
    # ------------------------------------------------------------------
    def submit(
        self,
        text: str | Message,
        system: str | None = None,
        restore: Any = None,
    ) -> RunHandle | None:
        """Start an agent run in a worker thread; return its handle.

        TEXT may be a plain string (parsed for ``@file`` references
        here) or a pre-built Message (slash commands that already
        parsed their kickoff's ``@file`` references pass the Message
        directly).  SYSTEM overrides the session's system prompt for
        this run only.  RESTORE (if given) runs when the run finishes
        — used by slash commands to put back state they borrowed
        (e.g. project_dir).

        Returns None when the text was only failed ``@file``
        references with nothing else to send (the view surfaces the
        validation errors from the handle).
        """
        user_msg, display_text, errors, warnings = self._build_user_message(text)
        if user_msg is None:
            return None
        # A new top-level run starts here: drop any todo list left over
        # from a previous run so a finished task's todos don't stay
        # pinned into the next task.
        self.clear_todos()
        # A new top-level run starts here: invalidate any worker still
        # unwinding from a previous run — from this point on it is stale
        # and must never touch shared state.  Bump before clearing the
        # event so there is no instant where an old worker sees "not
        # cancelled".
        self.session.run_generation += 1
        self.session.cancel_event.clear()
        seq = self.session.run_generation
        worker = threading.Thread(
            target=self._run_worker,
            args=(user_msg, seq, system, restore),
            daemon=True,
        )
        worker.start()
        return RunHandle(
            worker=worker,
            seq=seq,
            display_text=display_text,
            errors=errors,
            warnings=warnings,
        )

    def _build_user_message(
        self, text: str | Message
    ) -> tuple[Message | None, str, list[AttachmentError], list[str]]:
        """Build the user message for a run from raw input.

        Returns ``(user_msg, display_text, errors, warnings)``;
        ``user_msg`` is None when the input was only failed ``@file``
        references with nothing else to send.
        """
        if isinstance(text, Message):
            cleaned_text = text.text()
            has_images = isinstance(text.content, list) and any(
                isinstance(p, ImagePart) for p in text.content
            )
            warnings = self._image_warnings(has_images)
            return text, cleaned_text.strip() or "(attachment)", [], warnings
        cleaned_text, attachments, errors = parse_at_references(text, str(self.project_dir))
        if errors and not attachments and not cleaned_text.strip():
            # All references failed and nothing else to send
            return None, "", errors, []
        # Build the user message: multimodal if there are attachments,
        # plain text otherwise (preserving the existing text-only path).
        if attachments:
            parts: list[Any] = []
            if cleaned_text.strip():
                parts.append(TextPart(text=cleaned_text))
            for att in attachments:
                parts.append(att.part)
            user_msg = Message(role="user", content=parts)
            # The display text is the user-facing text (without image data)
            display_text = (
                cleaned_text.strip()
                or f"({len(attachments)} image attachment{'s' if len(attachments) != 1 else ''})"
            )
            has_images = any(isinstance(a.part, ImagePart) for a in attachments)
        else:
            user_msg = Message(role="user", content=cleaned_text)
            display_text = cleaned_text
            has_images = False
        return user_msg, display_text, errors, self._image_warnings(has_images)

    def _image_warnings(self, has_images: bool) -> list[str]:
        """Warnings when the active model cannot see attached images.

        The attachment will be silently stripped before the request is
        sent (see Client._payload), so the user should know the image
        never reaches the model.
        """
        if has_images and not self.supports_image_input:
            return [
                f"model {self.model} does not support image input — "
                "image attachment(s) will be ignored"
            ]
        return []

    def _run_worker(
        self,
        user_msg: Message,
        seq: int,
        system: str | None = None,
        restore: Any = None,
    ) -> None:
        """Worker-thread body of a submitted run: drive the agent loop.

        Only the current run (``seq == run_generation``) may update
        shared state: a stale worker (a newer run started) must not
        clobber the next run.  A cancelled run with no successor is
        still current, so it adopts its salvaged partial history and
        the interrupted turn is not lost (the seq check is the
        staleness guard; the cancel event no longer blocks the
        adoption).
        """
        try:
            self.conversation_history.append(user_msg)
            run_agent_loop(
                self.session,
                messages=list(self.conversation_history),
                top_level=True,
                system=system or self.system_prompt,
            )
            if seq == self.session.run_generation and self.session.last_messages:
                self.conversation_history = list(self.session.last_messages)
        except Exception as e:  # noqa: BLE001
            if seq == self.session.run_generation:
                self.session.log(f"agent error: {e}")
        finally:
            if seq == self.session.run_generation:
                # the run is done: the final assistant message is now
                # part of the conversation history, so the view can drop
                # its live stream buffer (otherwise the same text
                # renders twice — stream row + history row)
                self.session.notify("run_finished")
                if restore is not None:
                    restore()
