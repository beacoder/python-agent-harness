"""Shared session-restore logic for the TUI's /restore and headless --restore.

Both front ends resolve a saved session the same way (path, title
substring, or latest), parse it the same way, and apply it to a
``Controller`` the same way: repoint the session store, restore the
saved agent and model, and replace the conversation history.  Only the
reporting differs, so callers pass a ``report(level, message)``
callback ("ok" for confirmations, "warning" for problems).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..core.models import Message
from ..io import persistence

if TYPE_CHECKING:
    from .controller import Controller

AGENT_KEY = "python-agent-harness--agent"
MODEL_KEY = "python-agent-harness--model"
PROJECT_KEY = "python-agent-harness--project-dir"
ROUND_TIMES_KEY = "python-agent-harness--round-times"

Report = Callable[[str, str], None]


@dataclass
class SavedSession:
    """A parsed session file: its path, metadata and conversation."""

    path: str
    meta: dict[str, str] = field(default_factory=dict)
    messages: list[Message] = field(default_factory=list)

    @property
    def round_times(self) -> list[float]:
        """Persisted round timestamps; [] when absent or malformed."""
        raw = self.meta.get(ROUND_TIMES_KEY)
        if not raw:
            return []
        try:
            return [float(x) for x in raw.split()]
        except ValueError:
            return []


def resolve_session_path(spec: str) -> str | None:
    """Map SPEC (empty/"latest"/"--latest", a file path, or a title
    substring) to a session file path, or None when nothing matches.

    The result is not guaranteed to exist: callers check ``isfile``.
    """
    if not spec or spec in ("--latest", "latest"):
        return persistence.SessionPersistence.latest_session()
    expanded = os.path.expanduser(spec)
    if os.path.isfile(expanded):
        return expanded
    return persistence.find_session_by_title(spec)


def load_saved_session(path: str) -> SavedSession:
    """Read and parse the session file at PATH.  Raises OSError.

    surrogateescape mirrors ``SessionPersistence.save``: the saved body
    may carry lone surrogates (non-UTF-8 file content), which a strict
    decode would reject.
    """
    with open(path, encoding="utf-8", errors="surrogateescape") as fh:
        text = fh.read()
    store = persistence.SessionPersistence
    meta = store.parse_metadata(text)
    messages = persistence.parse_saved_body(store.strip_metadata(text))
    return SavedSession(path=path, meta=meta, messages=messages)


def restore_agent(controller: Controller, agent: str | None, report: Report) -> None:
    """Switch to the saved AGENT; fall back to the default agent (so no
    stale tool exclusions linger) when it no longer exists."""
    if not agent:
        return
    try:
        success, msg = controller.switch_agent(agent)
        if success:
            report("ok", msg)
        else:
            controller.switch_agent("default")
            report("warning", msg)
    except Exception as e:  # noqa: BLE001 - a bad agent must not abort the restore
        report("warning", f"could not restore agent: {e}")


def restore_model(controller: Controller, model: str | None, report: Report) -> None:
    """Switch back to the saved raw MODEL name through a profile.

    Profile switching also fixes base_url/api_key/temperature, which a
    bare model-name change would leave pointing at the wrong endpoint.
    Order: the "default" pseudo-profile when MODEL is the session-start
    model, else MODEL as a profile name, else a profile whose model
    matches.  Without a match the current model stays active.
    """
    if not model or model == controller.model:
        return
    try:
        if model == controller.llm_settings.get("model"):
            _, msg = controller.switch_model("default")
            report("ok", msg)
            return
        success, msg = controller.switch_model(model)
        if not success:
            profile = next(
                (name for name, p in controller.model_profiles.items() if p.get("model") == model),
                None,
            )
            if profile:
                success, msg = controller.switch_model(profile)
        if success:
            report("ok", msg)
        else:
            report("warning", f"model {model!r} has no matching profile, keeping current model")
    except Exception as e:  # noqa: BLE001 - a bad model must not abort the restore
        report("warning", f"could not restore model: {e}")


def apply_saved_session(controller: Controller, saved: SavedSession, report: Report) -> None:
    """Make SAVED the controller's current session.

    The conversation history is replaced in BOTH places the runtime
    reads it: ``conversation_history`` (what the next run sends) and
    ``last_messages`` (what the views render).  ``run_generation`` is
    bumped so a worker still winding down from a cancelled run cannot
    clobber the restored history with its salvaged one.
    """
    store = controller.store
    store.round_times = saved.round_times
    store.file_path = saved.path
    title = persistence.title_from_filename(saved.path)
    if title:
        store.title = title
    restore_agent(controller, saved.meta.get(AGENT_KEY), report)
    restore_model(controller, saved.meta.get(MODEL_KEY), report)
    controller.run_generation += 1
    controller.conversation_history = list(saved.messages)
    controller.last_messages = list(saved.messages)
    controller.clear_todos()
