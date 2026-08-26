"""Slash command handling for the TUI.

Contains the CommandMixin with all slash command dispatch, session
command execution (/init /review /explain), /model switching,
/compact, /summary, /sessions, /restore, and session body parsing.
"""

from __future__ import annotations

import os
import shlex
import threading
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.live import Live
from prompt_toolkit.patch_stdout import patch_stdout

from .. import config
from ..commands import find_command
from ..models import Message
from ..persistence import (
    SessionPersistence,
    escape_role_headers,
    split_role_header,
    title_from_filename,
    unescape_role_header,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..session import Session


class CommandMixin:
    """Slash command methods for the TUI.

    Expects the host class to provide: ``session``, ``console``,
    ``conversation_history``, ``_history_dirty``, ``_data_event``,
    ``agent_running``, ``status``, ``_current_tool``, ``_start_agent``,
    ``_status_bar``, ``_flush``, ``_render_frame``.
    """

    if TYPE_CHECKING:
        session: Session
        console: Console
        conversation_history: list[Message]
        _history_dirty: bool
        _data_event: threading.Event
        agent_running: bool
        status: str
        _current_tool: str
        _round_times: list[float]

        def _start_agent(
            self,
            text: str,
            system: str | None = None,
            restore: Callable[[], None] | None = None,
        ) -> None: ...

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
        if cmd == "/plan":
            self.session.switch_to_plan()
            self.console.print(
                "[yellow]Plan mode — read-only; only the plan file is writable.[/yellow]"
            )
        elif cmd == "/build":
            self.session.switch_to_build()
            self.console.print("[green]Build mode.[/green]")
        elif cmd == "/compact":
            self._run_compact()
        elif cmd == "/save":
            path = self.session.store.save(self._conversation_text())
            self.console.print(f"saved: {path}")
        elif cmd == "/summary":
            self._run_summary()
        elif cmd in ("/init", "/review", "/explain"):
            self._run_slash_command(cmd[1:], arg)
        elif cmd == "/sessions":
            self._run_sessions()
        elif cmd == "/restore":
            self._run_restore(arg)
        elif cmd == "/clear":
            # Replacing the conversation is a new generation: invalidate any
            # worker still winding down from a cancelled run, or its
            # salvaged-history commit would resurrect what we just wiped.
            self.session.run_generation += 1
            self.conversation_history = []
            self.session.last_messages = []
            self.session.clear_todos()
            self.console.print("[yellow]Conversation history cleared.[/yellow]")
        elif cmd == "/model":
            self._run_model_command(arg)
        elif cmd == "/help":
            self.console.print(
                "/plan /build /init /review /explain /compact "
                "/save /summary /sessions /restore /clear /model /exit\n"
                "/init [project] [--extra TEXT]       create/update AGENTS.md\n"
                "/review [project] [commit|branch|PR] review code changes\n"
                "/explain [project] [target]          explain code\n"
                "/sessions                            list saved sessions\n"
                "/restore [path | title | --latest | latest]   restore a saved session\n"
                "/model [name]                        switch LLM model profile\n"
                "Ctrl-C cancels the current execution (app stays open); "
                "Ctrl-D or /exit quits.",
                markup=False,
            )
        else:
            self.console.print(f"unknown command: {cmd}")
        return False

    # ------------------------------------------------------------------
    # command slash commands (/init /review /explain)
    # ------------------------------------------------------------------
    @staticmethod
    def _split_args(arg: str) -> list[str]:
        """Split a slash-command argument string (shell-like quoting)."""
        try:
            return shlex.split(arg)
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
        """Run a SessionCommand (/init /review /explain) in this session.

        The command's prompt replaces the system prompt for this run
        only; the output streams into the conversation panel and stays
        in history, so the user can follow up on the result.  When a
        different project is given, the session's project dir is
        borrowed for the run (tool cwd) and restored afterwards.
        """
        cmd = find_command(name)
        if cmd is None:
            self.console.print(f"[yellow]unknown command: /{name}[/yellow]")
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
            project_dir=project or self.session.project_dir, extra=extra
        )
        if self.conversation_history or self.session.last_messages:
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
            cwd, prompt, context_path=self.session._configured_context_path
        )
        prev_project = self.session.project_dir
        if cwd != prev_project:
            self.session.project_dir = cwd

            def _restore() -> None:
                # idempotent: only undo OUR borrow, never clobber a
                # newer run's borrow (or a restore already performed)
                if self.session.project_dir == cwd:
                    self.session.project_dir = prev_project

            restore = _restore
        else:
            restore = None
        if not cmd.allow_planexit:
            # init/review: all tools except PlanExit — hide it for the
            # run (sub-agents share the session registry, so they are
            # covered too) and put it back when the run finishes.
            from ..commands import hide_planexit

            planexit_restore = hide_planexit(self.session)
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
        self.console.print(f"[cyan]/{name}: {kickoff.strip()}[/cyan]")
        self._start_agent(kickoff, system=system, restore=restore)

    def _conversation_text(self) -> str:
        msgs = self.session.last_messages or []
        parts = []
        for m in msgs:
            # escaped: see persistence.escape_role_headers
            body = escape_role_headers(m.text())
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
    ) -> None:
        """Run *worker* in a background thread with status bar updates.

        Sets the status bar to *status_text*, renders the spinner while
        the worker is in flight, and handles KeyboardInterrupt with
        *cancel_message*.  Used by /compact and /summary to avoid
        duplicating the thread + Live boilerplate.
        """
        self.status = status_text
        self._current_tool = ""
        self.agent_running = True
        self._data_event.clear()

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        try:
            if self.console.is_dumb_terminal:
                while thread.is_alive():
                    self._data_event.wait(timeout=0.1)
                    self._data_event.clear()
                    self.console.print(self._status_bar())
                    self._flush()
            else:
                with Live(
                    self._status_bar(),
                    console=self.console,
                    refresh_per_second=30,
                    screen=False,
                ) as live:
                    while thread.is_alive():
                        self._data_event.wait(timeout=0.1)
                        self._data_event.clear()
                        live.update(self._status_bar())
                        self._flush()
        except KeyboardInterrupt:
            self.console.print(f"\n[dim]{cancel_message}[/dim]")
            self._flush()
        finally:
            self.agent_running = False
            self.status = ""
            self._data_event.set()

    def _run_compact(self) -> None:
        """Compact the current conversation directly."""
        result: dict[str, Any] = {}

        def worker() -> None:
            try:
                ok, msg = self.session.compact_conversation()
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
            self.conversation_history = list(self.session.last_messages)
            self._history_dirty = True
        self.console.print(msg)

    def _refresh_model_profiles(self) -> None:
        """Re-read the config file's ``models`` section so profiles
        added/removed while the TUI is running show up on the next
        /model call.  A malformed config keeps the last loaded set."""
        try:
            self.session.model_profiles = config.load_models_config(self.session.config_path)
        except ValueError as e:
            self.console.print(f"[red]{e}[/red]")

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
        return ["default", *sorted(self.session.model_profiles.keys())]

    def _model_switch_by_name(self, name: str) -> None:
        """Switch to a named profile (or ``default``) and report the outcome."""
        success, msg = self.session.switch_model(name)
        if success:
            self.console.print(f"[green]{msg}[/green]")
            self._data_event.set()
        else:
            self.console.print(f"[red]{msg}[/red]")

    def _run_model_command(self, arg: str) -> None:
        """Handle /model command for switching LLM profiles."""
        self._refresh_model_profiles()
        if not arg:
            # List all models: the original default plus every profile
            # currently in the config file
            all_names = self._model_list_names()
            current_model = self.session.model or "(unknown)"
            default_model = (self.session.llm_settings or {}).get("model") or current_model
            default_base_url = (self.session.llm_settings or {}).get(
                "base_url"
            ) or self.session.client.base_url

            self.console.print("\n[bold cyan]Available model profiles:[/bold cyan]")
            if not self.session.model_profiles:
                self.console.print(
                    "[yellow]  (none configured — add a 'models' section to use /model)[/yellow]"
                )
            for idx, name in enumerate(all_names, 1):
                if name == "default":
                    marker = " *" if current_model == default_model else ""
                    self.console.print(
                        f"  [cyan]{idx})[/cyan] default{marker} — "
                        f"{default_model} @ {default_base_url}"
                    )
                else:
                    profile = self.session.model_profiles[name]
                    model_name = profile.get("model", "(inherited)")
                    base_url = profile.get("base_url", "(inherited)")
                    marker = " *" if model_name == current_model else ""
                    self.console.print(
                        f"  [cyan]{idx})[/cyan] {name}{marker} — {model_name} @ {base_url}"
                    )

            total_count = len(all_names)
            self.console.print(f"\n[dim]Current: {current_model}[/dim]")
            self.console.print(
                f"[dim]Type a number (1-{total_count}) to switch, or enter a model name directly[/dim]\n"
            )

            # Prompt user for selection
            try:
                with patch_stdout():
                    selection = self.prompt_session.prompt("Select model: ").strip()
                if not selection:
                    return

                # Check if selection is a number
                if selection.isdigit():
                    idx = int(selection) - 1
                    if 0 <= idx < len(all_names):
                        selected = all_names[idx]
                        if selected == "default" and current_model == default_model:
                            self.console.print("[yellow]Already using this model.[/yellow]")
                        else:
                            self._model_switch_by_name(selected)
                    else:
                        self.console.print(
                            f"[red]Invalid selection: {selection}. Choose 1-{len(all_names)}[/red]"
                        )
                else:
                    # Treat as model name
                    self._model_switch_by_name(selection)
            except EOFError:
                pass
            except KeyboardInterrupt:
                self.console.print("\n[dim]cancelled[/dim]")
            return

        # Check if arg is a number
        if arg.strip().isdigit():
            all_names = self._model_list_names()
            idx = int(arg.strip()) - 1
            if 0 <= idx < len(all_names):
                selected = all_names[idx]
                default_model = (self.session.llm_settings or {}).get("model") or (
                    self.session.model or ""
                )
                if selected == "default" and (self.session.model or "") == default_model:
                    self.console.print("[yellow]Already using this model.[/yellow]")
                else:
                    self._model_switch_by_name(selected)
            else:
                self.console.print(
                    f"[red]Invalid selection: {arg}. Choose 1-{len(all_names)}[/red]"
                )
            return
        # Switch to named model
        self._model_switch_by_name(arg)

    def _run_summary(self) -> None:
        """Append a summary of the conversation (tools disabled)."""
        result: dict[str, str] = {}

        def worker() -> None:
            try:
                result["msg"] = self.session.summarize_conversation()
            except Exception as e:  # noqa: BLE001 - surfaced to the user
                result["msg"] = f"Summary failed: {e}"

        self._run_with_status(
            worker,
            status_text=" ⏳ summarizing",
            cancel_message="summary cancelled — the result may still be appended",
        )
        msg = result.get("msg", "Summary failed: unknown error.")
        self.conversation_history = list(self.session.last_messages)
        self._history_dirty = True
        if msg == "Summary appended.":
            last_msg = self.session.last_messages[-1]
            if last_msg.role == "assistant" and last_msg.content:
                self.console.print(last_msg.content)
            else:
                self.console.print(msg)
        else:
            self.console.print(msg)

    def _run_sessions(self) -> None:
        """List saved sessions with metadata."""
        files = SessionPersistence.list_sessions()
        if not files:
            self.console.print("[dim]no saved sessions[/dim]")
            return
        for f in files:
            try:
                with open(f, encoding="utf-8") as fh:
                    text = fh.read()
            except OSError:
                continue
            meta = SessionPersistence.parse_metadata(text)
            basename = os.path.basename(f)
            model = meta.get("gptel-model", "?")
            project = meta.get("python-agent-harness--project-dir", "?")
            self.console.print(f"  {basename:50s}  model={model:20s}  project={project}")

    def _run_restore(self, arg: str) -> None:
        """Restore a saved session into the current TUI.

        Usage: /restore <path>  or  /restore --latest  or  /restore latest
               or  /restore <title>
        Loads the conversation history so the user can continue from
        where they left off.  When the argument is not a file path,
        it is matched as a substring against session filenames/titles
        (case-insensitive).
        """
        path: str | None = None
        if not arg or arg in ("--latest", "latest"):
            path = SessionPersistence.latest_session()
        elif os.path.isfile(arg):
            path = arg
        else:
            # Try title-based matching: find sessions whose filename
            # contains the argument as a case-insensitive substring
            path = self._find_session_by_title(arg)
        if not path:
            self.console.print(
                "[yellow]no session found "
                "(use /restore <path>, /restore <title>, "
                "/restore --latest, or /restore latest)[/yellow]"
            )
            return
        if not os.path.isfile(path):
            self.console.print(f"[red]file not found: {path}[/red]")
            return
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        except OSError as e:
            self.console.print(f"[red]cannot read {path}: {e}[/red]")
            return
        meta = SessionPersistence.parse_metadata(text)
        body = SessionPersistence.strip_metadata(text)
        # Rebuild conversation history from the saved markdown format
        messages = self._parse_saved_body(body)
        # Round timestamps are persisted in the metadata block; restore
        # them so the dump separators keep their HH:MM:SS times.
        round_times = meta.get("python-agent-harness--round-times")
        if round_times:
            try:
                self._round_times = [float(x) for x in round_times.split()]
            except ValueError:
                self._round_times = []
        else:
            self._round_times = []
        self.session.store.round_times = list(self._round_times)
        # Update the session store to point at the restored file
        self.session.store.file_path = path
        title = title_from_filename(path)
        if title:
            self.session.store.title = title
        # Replace conversation history: a new generation.  Invalidate any
        # worker still winding down from a cancelled run so its salvaged
        # history can't clobber the restored session.
        self.session.run_generation += 1
        self.conversation_history = messages
        self.session.last_messages = list(messages)
        self.session.clear_todos()
        self._history_dirty = True
        model = meta.get("gptel-model", "?")
        project = meta.get("python-agent-harness--project-dir", "?")
        self.console.print(
            f"[green]restored:[/green] {os.path.basename(path)} "
            f"(model={model}, project={project}, {len(messages)} messages)"
        )

    @staticmethod
    def _parse_saved_body(body: str) -> list[Message]:
        """Parse a saved session body back into Message objects.

        The save format is markdown with **role**: content blocks
        separated by blank lines.

        ``tool`` blocks are dropped: the saved markdown does not keep
        ``tool_call_id``/``name`` (assistant tool calls are flattened to
        plain text), so a restored ``role="tool"`` message would form an
        API-invalid payload (a tool message with no preceding assistant
        ``tool_calls``).  The following assistant reply already
        summarizes the results, so dropping them loses no essential
        context.

        Body lines that merely look like a block header are escaped by
        the renderer (see `escape_role_headers`) and unescaped here, so
        a message quoting this format no longer splits into extra
        messages.  Sessions saved before escaping existed can still
        split — that ambiguity is in the file, not in this parser.
        """
        messages: list[Message] = []
        current_role: str | None = None
        current_lines: list[str] = []

        for line in body.splitlines():
            # Check for a role header: **user**: ... or **assistant**: ...
            header = split_role_header(line)
            if header is not None:
                role, rest = header
                # Save the previous block (tool blocks are dropped:
                # their tool_call_id/name were not persisted)
                if current_role is not None and current_role != "tool":
                    content = "\n".join(current_lines).strip()
                    if content:
                        messages.append(Message(role=current_role, content=content))
                current_role = role
                current_lines = [unescape_role_header(rest)]
                continue
            current_lines.append(unescape_role_header(line))

        # Don't forget the last block (tool blocks are dropped)
        if current_role is not None and current_role != "tool":
            content = "\n".join(current_lines).strip()
            if content:
                messages.append(Message(role=current_role, content=content))

        return messages

    @staticmethod
    def _find_session_by_title(query: str) -> str | None:
        """Find a session file by title substring (case-insensitive).

        Matches against the full filename, the filename without .md,
        and the derived title.  Returns the most recent match, or None.
        """
        query_lower = query.lower()
        # Strip .md from query if present, for cleaner substring matching
        query_stem = query_lower[:-3] if query_lower.endswith(".md") else query_lower
        files = SessionPersistence.list_sessions()  # already sorted by mtime desc
        for f in files:
            basename = os.path.basename(f)
            basename_lower = basename.lower()
            # Exact basename match (with or without .md)
            if basename_lower == query_lower or basename_lower == query_lower + ".md":
                return f
            # Substring match against filename (minus .md)
            name_part = basename[:-3] if basename.endswith(".md") else basename
            if query_stem in name_part.lower():
                return f
            # Match against derived title (dashes → spaces)
            title = title_from_filename(f)
            if title and query_stem in title.lower():
                return f
        return None
