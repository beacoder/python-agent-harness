"""Tool base classes, the tool registry, and shared tool helpers."""

from __future__ import annotations

import contextlib
import os
import stat
import threading
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..core.models import ToolSpec


def atomic_write_text(path: str, content: str) -> None:
    """Replace PATH's contents with CONTENT atomically.

    THE single write path for every tool that rewrites a file (``Edit``,
    ``Insert``, ``Write``, and the pure-Python diff applier).  It lives
    here, in the module every tool already imports, so the four callers
    cannot drift apart -- and because ``base`` imports nothing from
    ``tools``, putting it here is also the only placement that avoids a
    circular import: ``filesystem.py`` re-imports ``edit``/``write``/
    ``insert`` at its bottom, so a helper defined *there* would make
    ``edit`` import a half-initialised ``filesystem``.

    A plain ``open(path, "w")`` TRUNCATES the file before writing, so a
    write that fails partway through -- ENOSPC, a quota, an I/O error,
    the process being killed -- left the file empty and the user's
    content unrecoverable while the tool returned a tidy "Error: ..."
    string.  Instead the new content goes to a sibling temp file which is
    then renamed over the target, mirroring the tmp-write + ``os.replace``
    discipline in ``SessionPersistence.save``.  ``os.replace`` is atomic:
    a reader sees either the complete old file or the complete new one,
    and ANY failure before it leaves PATH untouched.  The temp file is
    removed on the error path.

    Deliberate details:

    - ``errors="surrogateescape"`` and ``newline=""`` match the tools'
      read side, so invalid UTF-8 bytes (Latin-1, GBK, ...) round-trip
      and the content is written byte-for-byte instead of having "\\n"
      translated to ``os.linesep`` on Windows.
    - The temp name carries a random suffix rather than a fixed ".tmp":
      concurrent sub-agents can edit the same path, and a shared name
      would let one writer rename the other's half-written file into
      place.
    - An existing file's permission bits are copied onto the
      replacement, so editing an executable script does not drop its
      ``+x``.  A file that does not exist yet keeps the umask-derived
      permissions a plain ``open(path, "w")`` would have produced.  Only
      the "no such file" case is tolerated: any other stat/chmod failure
      propagates rather than silently shipping the wrong mode.

    There is deliberately no ``fsync``: the failures this guards against
    are process-level (a failed write, a kill, Ctrl-C), and in all of
    them the rename never happens.  Surviving a power loss in the window
    between write and rename would need an fsync here and on the parent
    directory, which ``SessionPersistence.save`` does not do either.

    Consequences of replacing the file rather than rewriting it in place,
    all inherent to the atomic-replace approach:

    - PATH must already be symlink-resolved.  ``os.replace`` onto a
      symlink overwrites the LINK with a regular file, where an in-place
      write would have followed it.  Every caller resolves the real path
      first (``Edit``/``Insert``/``Write`` via ``os.path.realpath`` on
      entry, ``_apply_section`` on its resolved target).
    - Hard links are broken: the new content lands on a new inode, so
      other links to the old inode keep the old content.
    - Owner/group, ACLs, SELinux labels and chattr flags are NOT carried
      over (only the mode bits are).  This matters mainly for an agent
      running as root over files owned by someone else.
    - On a crash between the write and the rename the temp file survives
      as ``<name>.<hex>.tmp``; the original is still intact, but the
      stray file is visible to Glob/Grep and to ``git status``.

    Raises OSError on failure.  Note this needs write permission on the
    DIRECTORY, not just on the file -- the one behavioural difference
    from a truncating in-place write (GNU ``patch``, used by ``Edit``'s
    diff mode, has always had the same requirement).
    """
    try:
        mode: int | None = stat.S_IMODE(os.stat(path).st_mode)
    except FileNotFoundError:
        mode = None  # brand-new file: keep the umask default from write_text
    tmp = f"{path}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        Path(tmp).write_text(content, encoding="utf-8", errors="surrogateescape", newline="")
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    except OSError:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


class PendingToolResult:
    """Handle for an asynchronous tool result (mirrors ``:async t``).

    An async tool's ``run`` returns this handle instead of a string: it
    starts its background work (e.g. a spawned process) and returns
    immediately, then delivers the final result string later via
    ``deliver`` — so the wait never blocks the sequential tool loop.

    ``deliver`` is idempotent (first delivery wins, late duplicates are
    no-ops — mirroring the gptel-agent FSM's idempotent-result advice);
    ``wait`` blocks until the result has been delivered.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._result: str | None = None

    def deliver(self, result: str) -> None:
        if not self._event.is_set():
            self._result = result
            self._event.set()

    def wait(self) -> str:
        self._event.wait()
        return self._result or ""


@runtime_checkable
class ToolRuntime(Protocol):
    """The session surface a ``ToolContext`` proxies to.

    ``ToolContext`` needs only this slice of the full ``Session`` —
    a project directory, a cancel event, and a handful of callbacks
    for user questions, diff/todo recording, skill lookup, sub-agent
    delegation, and plan-mode exit.  Typing the context against this
    protocol (instead of ``Any``) documents the real dependency and
    lets a tool be exercised with a lightweight fake runtime, no full
    ``Session`` required::

        class FakeRuntime:
            project_dir = "/tmp"
            cancel_event = threading.Event()
            def ask_questions(self, questions): return "..."
            ...  # every ToolRuntime member (see below)

        ctx = ToolContext(FakeRuntime())
        tool.run(args, ctx)

    ``ToolContext`` still accepts ``None`` (its methods fall back to safe
    no-op defaults when there is no session).  For a non-``None`` runtime
    it proxies each call unconditionally (no ``hasattr`` probing): a
    complete runtime like the real ``Session`` always works, and a
    partial fake works too *as long as the tool under test only reaches
    for members it implements* — a missing member surfaces as a plain
    ``AttributeError`` rather than a silent default.  Test doubles that
    stand in for a full ``Session`` therefore implement every member and
    assert conformance via ``x: ToolRuntime = FakeThing()``.

    ``project_dir`` / ``cancel_event`` are declared as read-only
    properties (``ToolContext`` only ever reads them): that admits both
    a plain-attribute implementation (the real ``Session``) and a
    property-backed one (test doubles), whereas a plain mutable-attribute
    declaration would be invariant and reject a ``property``.
    """

    @property
    def project_dir(self) -> str: ...

    @property
    def cancel_event(self) -> threading.Event: ...

    @property
    def config_path(self) -> str | None: ...

    def ask_questions(self, questions: list[dict]) -> str: ...

    def record_diff(self, diff_text: str) -> None: ...

    def update_todos(self, todos: list[dict]) -> None: ...

    def find_skill(self, name: str) -> str | None: ...

    def run_subagent(self, description: str, prompt: str) -> str: ...

    def plan_exit(self) -> str: ...


class ToolContext:
    """Runtime context handed to tools.

    Tools may call back into the session for user questions,
    plan-mode checks, and sub-agent delegation.  All methods
    proxy to the session when present; defaults are safe no-ops.
    """

    def __init__(self, session: ToolRuntime | None = None) -> None:
        self.session = session

    @property
    def cwd(self) -> str:
        return self.session.project_dir if self.session else "."

    def ask_questions(self, questions: list[dict]) -> str:
        if self.session:
            return self.session.ask_questions(questions)
        return "Unanswered"

    def record_diff(self, diff_text: str) -> None:
        """Attach a unified diff to the currently-executing tool call."""
        if self.session:
            self.session.record_diff(diff_text)

    def update_todos(self, todos: list[dict]) -> None:
        if self.session:
            self.session.update_todos(todos)

    def find_skill(self, name: str) -> str | None:
        if self.session:
            return self.session.find_skill(name)
        return None

    def run_subagent(self, description: str, prompt: str) -> str:
        if self.session:
            return self.session.run_subagent(description, prompt)
        return f"Error: Task {description!r} returned an unexpected response — no session"

    def plan_exit(self) -> str:
        if self.session:
            return self.session.plan_exit()
        return "Not in plan mode; PlanExit has no effect.  Continue as normal."

    @property
    def cancel_event(self) -> threading.Event | None:
        """Session cancel event (set when the user presses Ctrl-C)."""
        if self.session:
            return self.session.cancel_event
        return None

    @property
    def config_path(self) -> str | None:
        """Path to the active config file (None = default resolution).

        Tools that read the config file (e.g. the LSP tool, which loads
        per-extension server overrides) use this so a session started
        with ``--config PATH`` reads the same file.
        """
        if self.session:
            return self.session.config_path
        return None


class Tool(ABC):
    name: str = ""
    description: str = ""
    # Detailed usage instructions injected into the system prompt for
    # this tool (when to use / not use / how).  An empty string means
    # no per-tool instructions are emitted (the ``description`` field
    # in the tool spec is the only guidance the model receives).
    instructions: str = ""
    # True for tools that only read state (Read, Glob, Grep, Skill):
    # when EVERY call in a round is readonly, the runner dispatches
    # them concurrently via a thread pool instead of one-at-a-time,
    # since none can depend on another's side effects.
    is_readonly: bool = False
    # NB: Tool is an ABC, not a dataclass, so this is a plain class-level
    # default (never mutated in place — every concrete tool overrides it
    # with its own schema).  It must be a real dict: a dataclasses.field()
    # sentinel here would silently become the "parameters" of any tool
    # that forgot to override it and then fail JSON serialization.
    parameters: dict[str, Any] = {}

    @abstractmethod
    def run(self, args: dict[str, Any], ctx: ToolContext) -> str | PendingToolResult:
        """Execute the tool and return the result string.

        Async tools return a ``PendingToolResult`` instead (see
        ``Bash``): the background work is spawned here and the final
        string is delivered later via ``PendingToolResult.deliver``.
        """

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


class Registry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._lock = threading.Lock()

    def register(self, tool: Tool) -> None:
        with self._lock:
            self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        with self._lock:
            self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self, names: list[str] | None = None) -> list[ToolSpec]:
        with self._lock:
            items = list(self._tools.items())
        wanted = set(names) if names is not None else {n for n, _ in items}
        return [t.spec() for name, t in items if name in wanted]

    def tool_instructions(self, names: list[str] | None = None) -> dict[str, str]:
        """Return ``{tool_name: instructions}`` for tools with non-empty
        ``instructions``.  When *names* is given, only those tools are
        included; otherwise every registered tool with instructions is
        returned.  Tools are returned in registration order.
        """
        with self._lock:
            items = list(self._tools.items())
        wanted = set(names) if names is not None else {n for n, _ in items}
        return {
            name: tool.instructions for name, tool in items if name in wanted and tool.instructions
        }

    def execute(self, name: str, args: dict[str, Any], ctx: ToolContext) -> str | PendingToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: unknown tool {name!r}"
        try:
            return tool.run(args, ctx)
        except Exception as e:  # noqa: BLE001 - errors become tool results
            return f"Error: tool {name} failed — {e}"
