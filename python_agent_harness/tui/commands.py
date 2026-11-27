"""Slash command handling for the TUI.

Contains the CommandMixin with all slash command dispatch, session
command execution (/init /review /explain), /model switching,
/compact, /summary, /sessions, /restore, and session body parsing.
"""

from __future__ import annotations

import contextlib
import os
import shlex
import threading
import time
from typing import TYPE_CHECKING, Any

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.live import Live
from rich.text import Text

from ..core.models import ImagePart, Message, TextPart
from ..entry.restore import (
    PROJECT_KEY,
    apply_saved_session,
    load_saved_session,
    resolve_session_path,
)
from ..io.attachments import image_placeholder, parse_at_references
from ..io.persistence import (
    SessionPersistence,
    escape_role_headers,
    parse_saved_body,
)
from ..prompts import RESERVED_AGENT_NAME, discover_agents
from ..session import config
from ..session.commands import find_command
from .input import _history_path
from .render import safe_markup, sanitize_terminal

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..entry.controller import Controller
    from ..llm.client import LLMClient


class _ProbeResults:
    """Thread-safe sink for the /model health probes.

    Probes run in parallel threads and record as each one finishes, so an
    interrupted listing keeps what was already checked.  A plain dict is
    not enough: the listing snapshots the results while a straggler may
    still be inserting, and an unguarded read of a mutating dict can raise
    "dictionary changed size during iteration" (or return a torn copy).

    It also tracks the probe clients so ``abort_all`` can wake the ones
    still blocked: Python threads cannot be killed, and a probe parked in
    ``recv`` would otherwise outlive the Ctrl-C until its timeout expired.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._results: dict[str, tuple[bool, str]] = {}
        self._clients: list[LLMClient] = []
        self._aborted = False

    def track(self, client: LLMClient) -> None:
        with self._lock:
            self._clients.append(client)
            aborted = self._aborted
        if aborted:
            # the cancel already happened: a probe that starts now would
            # otherwise run to its full timeout with nobody waiting on it
            with contextlib.suppress(Exception):  # best effort
                client.abort_probe()

    def untrack(self, client: LLMClient) -> None:
        """Drop a finished probe so a later cancel does not abort it again."""
        with self._lock, contextlib.suppress(ValueError):
            self._clients.remove(client)

    def record(self, name: str, outcome: tuple[bool, str]) -> None:
        with self._lock:
            self._results[name] = outcome

    def snapshot(self) -> dict[str, tuple[bool, str]]:
        with self._lock:
            return dict(self._results)

    def abort_all(self) -> None:
        """Abort every probe client still in flight (best effort).

        Uses ``abort_probe`` rather than ``abort``: a probe client is
        throwaway, so the pool rebuild ``abort`` performs would be pure
        latency (one CA-bundle load per profile) on the Ctrl-C path.
        """
        with self._lock:
            self._aborted = True
            clients = list(self._clients)
        for client in clients:
            with contextlib.suppress(Exception):
                client.abort_probe()

    def __len__(self) -> int:
        with self._lock:
            return len(self._results)


class CommandMixin:
    """Slash command methods for the TUI.

    Expects the host class to provide: ``_controller``, ``console``,
    ``conversation_history``, ``_history_dirty``, ``_data_event``,
    ``agent_running``, ``status``, ``lock``, ``_current_tool``, ``_start_agent``,
    ``_on_notify``, ``_status_bar``, ``_flush``, ``_render_frame``.
    """

    if TYPE_CHECKING:
        _controller: Controller
        console: Console
        prompt_session: PromptSession
        conversation_history: list[Message]
        _history_dirty: bool
        _data_event: threading.Event
        agent_running: bool
        status: str
        lock: threading.Lock
        _current_tool: str
        _round_times: list[float]

        def _start_agent(
            self,
            text: str | Message,
            system: str | None = None,
            restore: Callable[[], None] | None = None,
        ) -> None: ...

        def _on_notify(self, kind: str, data: Any = None) -> None: ...

        def _status_bar(self) -> Any: ...

        def _flush(self) -> None: ...

        def _render_frame(self) -> Any: ...

    # ------------------------------------------------------------------
    # slash commands
    # ------------------------------------------------------------------
    def _handle_slash(self, line: str) -> bool:
        cmd, _, arg = line.partition(" ")
        cmd = cmd.strip().lower()
        arg = arg.strip()
        if cmd == "/exit":
            return True
        # A command file replaces the builtin slash command of the same
        # name (configured directory first, then bundled), so /review,
        # /plan, /help, ... can be redefined by a .md file.  Nothing else
        # reaches this branch: find_command returns None for a name no
        # file claims and no builtin prompt command backs.
        if find_command(cmd[1:], self._controller.configured_command_path) is not None:
            self._run_slash_command(cmd[1:], arg)
            return False
        if cmd == "/plan":
            self._controller.switch_to_plan()
            self.console.print(
                "[yellow]Plan mode — read-only; only the plan file is writable.[/yellow]"
            )
        elif cmd == "/build":
            self._controller.switch_to_build()
            self.console.print("[green]Build mode.[/green]")
        elif cmd == "/compact":
            self._run_compact()
        elif cmd == "/save":
            path = self._controller.save(self._conversation_text())
            self.console.print(f"saved: {safe_markup(path)}")
        elif cmd == "/summary":
            self._run_summary()
        elif cmd == "/sessions":
            self._run_sessions()
        elif cmd == "/restore":
            self._run_restore(arg)
        elif cmd == "/clear":
            # Replacing the conversation is a new generation: invalidate any
            # worker still winding down from a cancelled run, or its
            # salvaged-history commit would resurrect what we just wiped.
            self._controller.clear_conversation()
            self.conversation_history = []
            self.console.print("[yellow]Conversation history cleared.[/yellow]")
        elif cmd == "/model":
            self._run_model_command(arg)
        elif cmd == "/agent":
            self._run_agent_command(arg)
        elif cmd == "/help":
            from ..session.commands import (
                load_custom_commands,
                unoverridable_command_names,
            )
            from .input import SLASH_COMMANDS

            configured_dir = self._controller.configured_command_path
            custom_names = {c.name for c in load_custom_commands(configured_dir)}
            builtin_names = [c.lstrip("/") for c in SLASH_COMMANDS]
            # A command file replaces the builtin of the same name, so that
            # name is listed once (in the custom line) instead of twice.
            builtin = " ".join(f"/{n}" for n in builtin_names if n not in custom_names)
            custom = " ".join(f"/{n}" for n in sorted(custom_names))
            overridden = sorted(n for n in custom_names if n in set(builtin_names))
            notes = ""
            if overridden:
                notes += "\noverrides builtin: " + ", ".join(f"{n}.md" for n in overridden)
            unoverridable = unoverridable_command_names(configured_dir)
            if unoverridable:
                notes += "\nignored (cannot override): " + ", ".join(
                    f"{n}.md" for n in unoverridable
                )
            explain_line = (
                "/explain [project] [target]          explain code\n"
                if "explain" in custom_names
                else ""
            )
            # A builtin replaced by a command file no longer has its own
            # usage: the file's prompt takes the arguments now.
            init_line = (
                ""
                if "init" in custom_names
                else "/init [project] [--extra TEXT]       create/update AGENTS.md\n"
            )
            review_line = (
                ""
                if "review" in custom_names
                else "/review [project] [commit|branch|PR] review code changes\n"
            )
            self.console.print(
                builtin
                + (f"\n{custom}" if custom else "")
                + "\n"
                + init_line
                + review_line
                + explain_line
                + "/sessions                            list saved sessions\n"
                "/restore [path | title | --latest | latest]   restore a saved session (conversation + agent + model)\n"
                "/model [name]                        switch LLM model profile\n"
                "/agent [name]                        switch agent system prompt\n"
                "Ctrl-C cancels the current execution (app stays open); "
                "Ctrl-D or /exit quits." + notes,
                markup=False,
            )
        else:
            self.console.print(f"unknown command: {safe_markup(cmd)}")
        return False

    # ------------------------------------------------------------------
    # command slash commands (/init /review /explain)
    # ------------------------------------------------------------------
    @staticmethod
    def _split_args(arg: str) -> list[str]:
        """Split a slash-command argument string (shell-like quoting).

        Backslashes are pre-escaped so POSIX shlex keeps them literal
        (``\\U`` etc. would otherwise be consumed as escape sequences
        and mangle paths containing backslashes).
        """
        try:
            return shlex.split(arg.replace("\\", "\\\\"))
        except ValueError:
            return arg.split()

    def _command_args(self, name: str, arg: str) -> tuple[str | None, str | None]:
        """Parse slash-command args into (project, extra).

        Positional order matches the CLI: [project] then the command's
        argument (commit/branch/PR for review, target for explain,
        --extra TEXT for init).  A lone first token that isn't an
        existing directory is treated as the command's argument instead
        of a project, so `/review main` reviews the branch `main` of the
        current project and `/explain client.py` explains `client.py`.
        """
        parts = self._split_args(arg)
        if not parts:
            return None, None
        if name == "init":
            project = None
            extra = None
            rest = parts
            if rest and rest[0] != "--extra":
                project = rest[0]
                rest = rest[1:]
            if rest:
                if rest[0] != "--extra":
                    return None, None
                extra = " ".join(rest[1:]) if len(rest) > 1 else None
            return project, extra
        first_is_dir = os.path.isdir(os.path.abspath(os.path.expanduser(parts[0])))
        if first_is_dir:
            return parts[0], " ".join(parts[1:]) or None
        return None, " ".join(parts)

    def _run_slash_command(self, name: str, arg: str) -> None:
        """Run a SessionCommand (/init /review /explain, or any command file
        that replaced a builtin) in this session.

        The command's prompt replaces the system prompt for this run
        only; the output streams into the conversation panel and stays
        in history, so the user can follow up on the result.  When a
        different project is given, the session's project dir is
        borrowed for the run (tool cwd) and restored afterwards.
        """
        cmd = find_command(name, self._controller.configured_command_path)
        if cmd is None:
            self.console.print(f"[yellow]unknown command: /{safe_markup(name)}[/yellow]")
            return
        project, extra = self._command_args(name, arg)
        if name == "explain" and project is None and not extra:
            self.console.print(
                "[yellow]/explain needs a target — e.g. /explain client.py "
                "or /explain the retry logic in client.py[/yellow]"
            )
            return
        if project:
            project = os.path.abspath(os.path.expanduser(project))
        cwd, prompt, kickoff = cmd.prepare(
            project_dir=project or self._controller.project_dir, extra=extra
        )
        # The command prompt is already-loaded text (${path}/$ARGUMENTS
        # substituted by prepare()), so inject tool guidance directly.
        # NOTE: excluded_tools is () — under an /agent profile with
        # exclude_tools: frontmatter, tool_specs() filters those tools
        # at runtime, so the prompt can document tools the model can't
        # call (deliberate: commands may escape the persona; revisit
        # prompt+runtime together if that changes).
        from ..prompts import inject_tool_guidance

        session = self._controller.session
        prompt = inject_tool_guidance(
            prompt,
            skill_dir=getattr(session, "_skill_dir", None),
            tool_instructions=(
                session.registry.tool_instructions()
                if getattr(session, "registry", None) is not None
                else None
            ),
        )
        if self.conversation_history or self._controller.last_messages:
            # The commands' kickoffs ("Proceed with the task described
            # in your instructions.") assume a fresh conversation: the
            # task lives only in the system prompt.  Mid-conversation
            # that reads as a continuation of the previous — already
            # finished — task, so the model keeps working on the old
            # one instead of the command's.  Anchor the new task by
            # naming the command (and target) and marking the earlier
            # conversation as background context only.
            target = f": {extra}" if extra else ""
            kickoff = (
                f"{kickoff.strip()}\n\n"
                f"This is a NEW /{name} request{target} — the messages "
                "above are background context from an earlier task; "
                "follow the NEW instructions in your system prompt."
            )
        # keep the project context + task-completion rules in front of
        # the command's prompt (the "actual agent prompt" for this run)
        from ..prompts import assemble_agent_prompt

        system = assemble_agent_prompt(
            cwd, prompt, context_path=self._controller.configured_context_path
        )
        prev_project = self._controller.project_dir
        if cwd != prev_project:
            self._controller.project_dir = cwd

            def _restore() -> None:
                # idempotent: only undo OUR borrow, never clobber a
                # newer run's borrow (or a restore already performed)
                if self._controller.project_dir == cwd:
                    self._controller.project_dir = prev_project

            restore = _restore
        else:
            restore = None
        if not cmd.allow_planexit:
            # init/review: all tools except PlanExit — hide it for the
            # run (sub-agents share the session registry, so they are
            # covered too) and put it back when the run finishes.
            from ..session.commands import hide_planexit

            planexit_restore = hide_planexit(self._controller.session)
            if planexit_restore is not None:
                prev_restore = restore
                state = {"done": False}

                def _restore() -> None:
                    if state["done"]:
                        return
                    state["done"] = True
                    if prev_restore is not None:
                        prev_restore()
                    planexit_restore()

                restore = _restore
        self.console.print(f"[cyan]/{safe_markup(name)}: {safe_markup(kickoff.strip())}[/cyan]")
        # Parse @file references in the kickoff (e.g. "/review @diff.patch"
        # or "/explain @client.py"): images become ImagePart attachments,
        # text files become TextPart attachments, and the @path token is
        # stripped from the text.  Validation errors are shown to the user.
        cleaned_kickoff, attachments, errors = parse_at_references(
            kickoff, str(self._controller.project_dir)
        )
        for err in errors:
            self.console.print(f"[red]@{safe_markup(err.path)}: {safe_markup(err.message)}[/red]")
        if attachments:
            parts: list[Any] = []
            if cleaned_kickoff.strip():
                parts.append(TextPart(text=cleaned_kickoff))
            for att in attachments:
                parts.append(att.part)
            kickoff_msg = Message(role="user", content=parts)
        else:
            kickoff_msg = Message(role="user", content=cleaned_kickoff)
        self._start_agent(kickoff_msg, system=system, restore=restore)

    def _conversation_text(self) -> str:
        msgs = self._controller.last_messages or []
        parts = []
        for m in msgs:
            # escaped: see persistence.escape_role_headers
            body = escape_role_headers(m.text())
            # Mark messages that contained image attachments (see
            # Session._conversation_text for details)
            if isinstance(m.content, list):
                image_count = sum(1 for p in m.content if isinstance(p, ImagePart))
                if image_count:
                    sources = [
                        loc
                        for p in m.content
                        if isinstance(p, ImagePart)
                        for loc in (p.path or p.url,)
                        if loc
                    ]
                    body = f"{image_placeholder(image_count, sources)}\n{body}"
            if body:
                parts.append(f"**{m.role}**: {body}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # direct commands (no LLM agent loop)
    # ------------------------------------------------------------------
    def _run_with_status(
        self,
        worker: Callable[[], None],
        *,
        status_text: str,
        cancel_message: str,
        transient: bool = False,
        restore_status: bool = False,
    ) -> bool:
        """Run *worker* in a background thread with status bar updates.

        Sets the status bar to *status_text*, renders the spinner while
        the worker is in flight, and handles KeyboardInterrupt with
        *cancel_message*.  Used by /compact, /summary and /model to avoid
        duplicating the thread + Live boilerplate.

        TRANSIENT erases the status bar once the worker finishes instead
        of leaving its last frame in the scrollback.  RESTORE_STATUS puts
        back the status that was showing before (e.g. a previous run's
        error) instead of clearing it.  Returns True when cancelled.
        """
        with self.lock:
            previous_status = self.status
            self.status = status_text
        self._current_tool = ""
        self.agent_running = True
        self._data_event.clear()

        cancelled = False
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        try:
            if self.console.is_dumb_terminal:
                # no cursor control: reprint only when the text changes,
                # not on every spinner tick
                printed = None
                while True:
                    with self.lock:
                        current = self.status
                    if current != printed:
                        self.console.print(self._status_bar())
                        self._flush()
                        printed = current
                    if not thread.is_alive():
                        break
                    self._data_event.wait(timeout=0.1)
                    self._data_event.clear()
            else:
                with Live(
                    self._status_bar(),
                    console=self.console,
                    refresh_per_second=30,
                    screen=False,
                    transient=transient,
                ) as live:
                    while thread.is_alive():
                        self._data_event.wait(timeout=0.1)
                        self._data_event.clear()
                        live.update(self._status_bar())
                        self._flush()
        except KeyboardInterrupt:
            cancelled = True
            self.console.print(f"\n[dim]{cancel_message}[/dim]")
            self._flush()
        finally:
            self.agent_running = False
            with self.lock:
                # only undo our own status: a background _on_log (e.g. an
                # MCP error) set during the run must survive
                if not restore_status:
                    self.status = ""
                elif self.status == status_text:
                    self.status = previous_status
            self._data_event.set()
        return cancelled

    def _run_compact(self) -> None:
        """Compact the current conversation directly."""
        result: dict[str, Any] = {}

        def worker() -> None:
            try:
                ok, msg = self._controller.compact_conversation()
                result["ok"] = ok
                result["msg"] = msg
            except Exception as e:  # noqa: BLE001 - surfaced to the user
                result["ok"] = False
                result["msg"] = f"Compaction failed: {e}"

        self._run_with_status(
            worker,
            status_text=" ⏳ compacting",
            cancel_message="compact cancelled — the result may still be applied",
        )
        ok = result.get("ok", False)
        msg = result.get("msg", "Compaction failed: unknown error.")
        if ok:
            # The shared conversation was replaced: sync the TUI's own
            # history too, or the next run would restart from the old
            # full conversation and immediately re-compact it.
            self.conversation_history = list(self._controller.last_messages)
            self._history_dirty = True
        self.console.print(Text(sanitize_terminal(str(msg))))

    def _refresh_model_profiles(self) -> None:
        """Re-read the config file's ``models`` section so profiles
        added/removed while the TUI is running show up on the next
        /model call.  A malformed config keeps the last loaded set."""
        try:
            self._controller.model_profiles = config.load_models_config(
                self._controller.config_path
            )
        except ValueError as e:
            self.console.print(f"[red]{safe_markup(e)}[/red]")

    def _model_list_names(self) -> list[str]:
        """Names shown by /model: the original default model (as
        ``default``) followed by every configured profile.

        ``default`` always restores the main ``llm`` settings the
        session started with, so the original model stays reachable
        after switching to profiles — and the listing count stays
        stable across switches.  Used by both the listing and the
        numbered-selection paths so the numbers always match what was
        displayed.
        """
        return ["default", *sorted(self._controller.model_profiles.keys())]

    def _model_switch_by_name(self, name: str) -> None:
        """Switch to a named profile (or ``default``).

        Only failures are reported: a successful switch already shows up
        in the input prompt's "model >" prefix, so an extra confirmation
        line is noise.
        """
        success, msg = self._controller.switch_model(name)
        if success:
            self._data_event.set()
        else:
            self.console.print(f"[red]{safe_markup(msg)}[/red]")

    def _resolve_profile_endpoint(self, name: str) -> tuple[str, str | None, str | None]:
        """Resolve the effective (base_url, api_key, model) for a profile name.

        Mirrors Session.switch_model's merge logic: start from the
        main llm settings, overlay the profile's non-None keys.
        """
        settings = dict(self._controller.llm_settings or {})
        for key in ("base_url", "api_key"):
            settings.setdefault(key, getattr(self._controller.client, key, None))
        settings.setdefault("model", self._controller.model)
        if name != "default":
            profile = self._controller.model_profiles.get(name, {})
            for key in ("base_url", "api_key", "model"):
                if profile.get(key) is not None:
                    settings[key] = profile[key]
        return (
            str(settings.get("base_url") or "").rstrip("/"),
            settings.get("api_key"),
            settings.get("model"),
        )

    @staticmethod
    def _model_health_budget(count: int) -> float:
        """Total wait for *count* profiles: the per-profile budget scaled by
        the profile count (never zero, so an empty list still short-circuits)."""
        return config.MODEL_HEALTH_TIMEOUT_PER_PROFILE * max(1, count)

    def _probe_model_availability(
        self,
        names: list[str],
        into: _ProbeResults | None = None,
    ) -> dict[str, tuple[bool, str]]:
        """Probe each profile's backend in parallel.

        Returns ``{name: (available, message)}``.  Each probe creates a
        temporary Client (to reuse CA-bundle resolution) and calls
        ``check_health()``.  The budget is
        ``config.MODEL_HEALTH_TIMEOUT_PER_PROFILE`` per profile, so a
        longer list waits proportionally longer for slow backends; httpx
        timeouts are per phase (and DNS is unbounded), so a shared
        deadline is what keeps the listing responsive.

        *into* lets a caller that can be interrupted (Ctrl-C) keep the
        probes that already finished: each probe records into it as it
        completes, so it holds partial results when this method is
        abandoned by its waiting thread.
        """
        from ..llm.client import Client

        timeout = self._model_health_budget(len(names))
        results = _ProbeResults() if into is None else into
        threads: list[threading.Thread] = []

        def probe(name: str) -> None:
            # any exception would kill the thread silently (row shows a bogus
            # "timeout") and print a traceback over the status bar
            try:
                outcome = probe_one(name)
            except Exception as e:  # noqa: BLE001 - reported in the row
                outcome = (False, f"error: {type(e).__name__}")
            results.record(name, outcome)

        def probe_one(name: str) -> tuple[bool, str]:
            base_url, api_key, model = self._resolve_profile_endpoint(name)
            if not base_url:
                return False, "no base_url"
            if not model:
                return False, "no model"
            verify = getattr(self._controller.client, "verify", None)
            client = Client(
                base_url=base_url,
                api_key=api_key,
                model=model,
                verify=verify,
                config_path=self._controller.config_path,
            )
            results.track(client)
            try:
                return client.check_health(timeout=timeout)
            finally:
                results.untrack(client)
                client.close()

        deadline = time.monotonic() + timeout
        for name in names:
            t = threading.Thread(target=probe, args=(name,), daemon=True)
            threads.append(t)
            t.start()
        for t in threads:
            t.join(timeout=max(0.0, deadline - time.monotonic()))
        # a probe past the deadline is still running: the fill-in below keeps
        # the returned mapping total without touching its shared collector
        snapshot = results.snapshot()
        return {name: snapshot.get(name, (False, "timeout")) for name in names}

    def _run_model_command(self, arg: str) -> None:
        """Handle /model command for switching LLM profiles."""
        self._refresh_model_profiles()
        if not arg:
            all_names = self._model_list_names()
            current_model = self._controller.model or "(unknown)"
            default_model = (self._controller.llm_settings or {}).get("model") or current_model
            default_base_url = (self._controller.llm_settings or {}).get(
                "base_url"
            ) or self._controller.client.base_url

            # probe before printing so the status bar's Live region never
            # interleaves with the listing; Ctrl-C abandons the wait but
            # keeps the rows probed so far
            progress = _ProbeResults()
            finished: dict[str, dict[str, tuple[bool, str]]] = {}

            def worker() -> None:
                # probes record into PROGRESS as they finish; only the final
                # (total) mapping is published here, so a worker outliving a
                # Ctrl-C never mutates what the rows read
                finished["health"] = self._probe_model_availability(all_names, into=progress)

            budget = self._model_health_budget(len(all_names))
            cancelled = self._run_with_status(
                worker,
                status_text=(
                    f" ⏳ Checking health of {len(all_names)} profile(s) "
                    f"(up to {budget:g}s, Ctrl-C to cancel)..."
                ),
                cancel_message="health check cancelled",
                transient=True,
                restore_status=True,
            )
            # a completed run reports every profile (unreached ones as
            # "timeout"); an interrupted one only the profiles that answered
            health = finished["health"] if "health" in finished else progress.snapshot()
            if cancelled:
                # wake the probes still blocked in recv: a thread cannot be
                # killed, so aborting their sockets is what stops them from
                # lingering until the budget expires
                progress.abort_all()
            if cancelled and len(health) < len(all_names):
                self.console.print(
                    f"[dim]{len(health)} of {len(all_names)} profile(s) checked "
                    "before the cancel — the rest are unprobed.[/dim]"
                )

            self.console.print("\n[bold cyan]Available model profiles:[/bold cyan]")
            if not self._controller.model_profiles:
                self.console.print(
                    "[yellow]  (none configured — add a 'models' section to use /model)[/yellow]"
                )

            for idx, name in enumerate(all_names, 1):
                if name == "default":
                    marker = " *" if current_model == default_model else ""
                    model_name = default_model
                    base_url = default_base_url
                else:
                    profile = self._controller.model_profiles[name]
                    model_name = profile.get("model", "(inherited)")
                    base_url = profile.get("base_url", "(inherited)")
                    marker = " *" if model_name == current_model else ""
                status = ""
                if name in health:
                    available, msg = health[name]
                    status = (
                        "  [green]✓[/green]" if available else f"  [red]✗ {safe_markup(msg)}[/red]"
                    )
                self.console.print(
                    f"  [cyan]{idx})[/cyan] {safe_markup(name)}{marker} — "
                    f"{safe_markup(model_name)} @ {safe_markup(base_url)}{status}"
                )

            total_count = len(all_names)
            self.console.print(f"\n[dim]Current: {safe_markup(current_model)}[/dim]")
            self.console.print(
                f"[dim]Type a number (1-{total_count}) to switch, or enter a model name directly[/dim]\n"
            )

            try:
                selection = input("Select model: ").strip()
                if not selection:
                    return

                if selection.isascii() and selection.isdigit():
                    idx = int(selection) - 1
                    if 0 <= idx < len(all_names):
                        selected = all_names[idx]
                        if selected == "default" and current_model == default_model:
                            self.console.print("[yellow]Already using this model.[/yellow]")
                        else:
                            self._model_switch_by_name(selected)
                    else:
                        self.console.print(
                            f"[red]Invalid selection: {safe_markup(selection)}. Choose 1-{len(all_names)}[/red]"
                        )
                else:
                    self._model_switch_by_name(selection)
            except EOFError:
                pass
            except KeyboardInterrupt:
                self.console.print("\n[dim]cancelled[/dim]")
            return

        if arg.strip().isascii() and arg.strip().isdigit():
            all_names = self._model_list_names()
            idx = int(arg.strip()) - 1
            if 0 <= idx < len(all_names):
                selected = all_names[idx]
                default_model = (self._controller.llm_settings or {}).get("model") or (
                    self._controller.model or ""
                )
                if selected == "default" and (self._controller.model or "") == default_model:
                    self.console.print("[yellow]Already using this model.[/yellow]")
                else:
                    self._model_switch_by_name(selected)
            else:
                self.console.print(
                    f"[red]Invalid selection: {safe_markup(arg)}. Choose 1-{len(all_names)}[/red]"
                )
            return
        self._model_switch_by_name(arg)

    def _run_summary(self) -> None:
        """Append a summary of the conversation (tools disabled)."""
        result: dict[str, str] = {}

        def worker() -> None:
            try:
                result["msg"] = self._controller.summarize_conversation()
            except Exception as e:  # noqa: BLE001 - surfaced to the user
                result["msg"] = f"Summary failed: {e}"

        self._run_with_status(
            worker,
            status_text=" ⏳ summarizing",
            cancel_message="summary cancelled — the result may still be appended",
        )
        msg = result.get("msg", "Summary failed: unknown error.")
        self.conversation_history = list(self._controller.last_messages)
        self._history_dirty = True
        if msg == "Summary appended.":
            last_msg = self._controller.last_messages[-1]
            if last_msg.role == "assistant" and last_msg.content:
                # model-written: Text (no markup parsing) + sanitized
                self.console.print(Text(sanitize_terminal(last_msg.text())))
            else:
                self.console.print(Text(sanitize_terminal(str(msg))))
        else:
            self.console.print(Text(sanitize_terminal(str(msg))))

    def _refresh_agent_profiles(self) -> None:
        """Re-discover agent prompt files (bundled + configured directory).

        Agent files may be added/removed while the TUI is running;
        this re-scans so new agents show up on the next /agent call.
        """
        self._discovered_agents = discover_agents(self._controller.configured_agent_path)

    def _agent_list_names(self) -> list[str]:
        """Names shown by /agent: ``default`` followed by every
        discovered agent (bundled + configured directory)."""
        return [RESERVED_AGENT_NAME, *sorted(self._discovered_agents.keys())]

    def _agent_switch_by_name(self, name: str) -> None:
        """Switch to a named agent (or ``default``).

        Only failures are reported: the status bar's " @agent" segment
        already reflects a successful switch.
        """
        success, msg = self._controller.switch_agent(name)
        if success:
            self._on_notify("agent-switched")
        else:
            self.console.print(f"[red]{safe_markup(msg)}[/red]")

    def _run_agent_command(self, arg: str) -> None:
        """Handle /agent command for switching agent system prompts."""
        self._refresh_agent_profiles()
        if not arg:
            all_names = self._agent_list_names()
            self.console.print("\n[bold cyan]Available agent profiles:[/bold cyan]")
            if not self._discovered_agents:
                self.console.print(
                    "  [yellow](none found — add .md files to the agents directory to use /agent)[/yellow]"
                )
            active_name = getattr(self._controller.session.store, "agent", None)
            for idx, name in enumerate(all_names, 1):
                if name == RESERVED_AGENT_NAME:
                    self.console.print(f"  [cyan]{idx})[/cyan] default — agent.md")
                else:
                    prompt_file = self._discovered_agents[name]
                    self.console.print(
                        f"  [cyan]{idx})[/cyan] {safe_markup(name)} — {safe_markup(prompt_file)}"
                    )
            if active_name and active_name not in self._discovered_agents:
                # The active prompt file disappeared while it was in use:
                # say so instead of silently listing nothing as active.
                self.console.print(
                    f"  [yellow](active agent {safe_markup(active_name)} no longer found on disk)[/yellow]"
                )
            total_count = len(all_names)
            self.console.print(
                f"\n[dim]Type a number (1-{total_count}) to switch, or enter an agent name directly[/dim]\n"
            )
            try:
                selection = input("Select agent: ").strip()
                if not selection:
                    return
                if selection.isascii() and selection.isdigit():
                    idx = int(selection) - 1
                    if 0 <= idx < len(all_names):
                        self._agent_switch_by_name(all_names[idx])
                    else:
                        self.console.print(
                            f"[red]Invalid selection: {safe_markup(selection)}. Choose 1-{len(all_names)}[/red]"
                        )
                else:
                    self._agent_switch_by_name(selection)
            except EOFError:
                pass
            except KeyboardInterrupt:
                self.console.print("\n[dim]cancelled[/dim]")
            return
        if arg.strip().isascii() and arg.strip().isdigit():
            all_names = self._agent_list_names()
            idx = int(arg.strip()) - 1
            if 0 <= idx < len(all_names):
                self._agent_switch_by_name(all_names[idx])
            else:
                self.console.print(
                    f"[red]Invalid selection: {safe_markup(arg)}. Choose 1-{len(all_names)}[/red]"
                )
            return
        self._agent_switch_by_name(arg)

    def _run_sessions(self) -> None:
        """List saved sessions with metadata."""
        files = SessionPersistence.list_sessions()
        if not files:
            self.console.print("[dim]no saved sessions[/dim]")
            return
        for f in files:
            try:
                # surrogateescape mirrors SessionPersistence.save (see there)
                with open(f, encoding="utf-8", errors="surrogateescape") as fh:
                    text = fh.read()
            except OSError:
                continue
            meta = SessionPersistence.parse_metadata(text)
            basename = os.path.basename(f)
            model = meta.get("python-agent-harness--model", "?")
            project = meta.get("python-agent-harness--project-dir", "?")
            body = SessionPersistence.strip_metadata(text)
            n_msgs = len(parse_saved_body(body))
            self.console.print(
                f"  {safe_markup(basename):50s}  model={safe_markup(model):20s}  project={safe_markup(project)}  {n_msgs} messages"
            )

    def _run_restore(self, arg: str) -> None:
        """Restore a saved session into the current TUI.

        Usage: /restore <path>  or  /restore --latest  or  /restore latest
               or  /restore <title>
        Loads the conversation history so the user can continue from
        where they left off.  When the argument is not a file path,
        it is matched as a substring against session filenames/titles
        (case-insensitive).
        """
        path = resolve_session_path(arg)
        if not path:
            self.console.print(
                "[yellow]no session found "
                "(use /restore <path>, /restore <title>, "
                "/restore --latest, or /restore latest)[/yellow]"
            )
            return
        if not os.path.isfile(path):
            self.console.print(f"[red]file not found: {safe_markup(path)}[/red]")
            return
        try:
            saved = load_saved_session(path)
        except OSError as e:
            self.console.print(f"[red]cannot read {safe_markup(path)}: {safe_markup(e)}[/red]")
            return

        def report(level: str, msg: str) -> None:
            # successful agent/model restores need no line of their own:
            # the status bar and the input prompt's "model >" prefix
            # already reflect them, only problems are worth reporting
            if level != "ok":
                self.console.print(f"[yellow]warning: {safe_markup(msg)}[/yellow]")

        apply_saved_session(self._controller, saved, report)
        # Round timestamps keep the dump separators' HH:MM:SS times.
        self._round_times = list(saved.round_times)
        self._history_dirty = True
        # The restored session belongs to the saved project: switch the
        # input history (Up/Down recall) to that project's file too, so
        # the recalled prompts match the restored conversation.
        saved_project = saved.meta.get(PROJECT_KEY)
        if saved_project:
            self._switch_input_history(saved_project)
        self.console.print(
            f"[green]restored:[/green] {safe_markup(os.path.basename(path))} "
            f"(project={safe_markup(saved_project or '?')}, {len(saved.messages)} messages)"
        )

    def _switch_input_history(self, project_dir: str) -> None:
        """Point the prompt session's Up/Down recall at PROJECT_DIR's
        input-history file.

        The FileHistory is bound once at Tui construction; swapping it
        here (session + default buffer) and resetting the buffer makes
        the next prompt load the restored project's history instead of
        the startup project's.
        """
        new_history = FileHistory(_history_path(project_dir))
        self.prompt_session.history = new_history
        buffer = self.prompt_session.default_buffer
        buffer.history = new_history
        # reset() cancels the pending history-load task and repopulates
        # the working lines from the new history on the next render
        buffer.reset()
