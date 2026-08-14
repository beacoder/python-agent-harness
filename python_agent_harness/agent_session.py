"""AgentSession: the runtime hub wiring tools, plan mode.

The session implements the ToolContext-facing API (sub-agents,
questions) and the agent-loop-facing API (client, calibrator, plan
mode, auto-save, notifications).  The TUI layer subclasses it to
provide interactive confirmations.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from collections.abc import Callable
from typing import Any

from . import config
from .client import Client
from .models import AgentMode
from .planmode import PlanMode
from .session_store import SessionStore
from .subagent import run_subagent
from .token_estimator import TokenCalibrator
from .tools import Registry, ToolContext
from .tools.base import PendingToolResult
from .tools.filesystem import cleanup_spooled_files


def find_skill_dir(project_dir: str, configured: str | None = None) -> str | None:
    """Locate the skill directory.

    If *configured* is set (from config file), use it directly.
    Otherwise search default locations (first match wins).
    """
    if configured and os.path.isdir(configured):
        return configured
    for cand in (
        os.path.join(os.path.expanduser("~"), ".emacs.d", "skills"),
        os.path.join(project_dir, "skills"),
    ):
        if os.path.isdir(cand):
            return cand
    return None


def find_context_dir(project_dir: str, configured: str | None = None) -> str | None:
    """Locate the default context directory.

    If *configured* is set (from config file), use it directly.
    Otherwise search default locations (first match wins).
    """
    if configured and os.path.isdir(configured):
        return configured
    for cand in (
        os.path.join(os.path.expanduser("~"), ".emacs.d", "contexts"),
        os.path.join(project_dir, "contexts"),
    ):
        if os.path.isdir(cand):
            return cand
    return None


class AgentSession:
    """One interactive agent session (a "buffer" in elisp terms)."""

    def __init__(
        self,
        project_dir: str,
        client: Client,
        model: str,
        backend: str = "OpenAI-compatible",
        system_prompt: str | None = None,
        subagent_system_prompt: str | None = None,
        temperature: float = config.TEMPERATURE,
        max_tokens: int | None = config.MAX_TOKENS,
        reasoning_effort: str | None = None,
        stream: bool = True,
        subagent_client: Client | None = None,
        subagent_temperature: float | None = None,
        subagent_max_tokens: int | None = None,
        subagent_reasoning_effort: str | None = None,
        subagent_stream: bool | None = None,
        tool_names: list[str] | None = None,
        registry: Registry | None = None,
        context_path: str | None = None,
        skill_path: str | None = None,
    ) -> None:
        self.project_dir = project_dir
        self.client = client
        self.model = model
        self.backend = backend
        self.system_prompt = system_prompt
        self.subagent_system_prompt = subagent_system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.stream = stream
        self.tools_enabled = True
        self.alive = True
        self._configured_context_path = context_path
        self._configured_skill_path = skill_path
        # Sub-agent LLM: a dedicated client (base_url/api_key/model/
        # timeout) and per-request options when a different LLM is
        # configured for sub-agents (mirrors gptel-agent-harness-
        # subagent-model/-backend); every unset option inherits the
        # main agent's value.  The sub-agent loop never uses this
        # client directly — each Agent tool invocation clones it
        # (see run_subagent) so concurrent sub-agents never share a
        # Client's pool/abort state.
        self.subagent_client = subagent_client or client
        self.subagent_temperature = (
            temperature if subagent_temperature is None else subagent_temperature
        )
        self.subagent_max_tokens = (
            max_tokens if subagent_max_tokens is None else subagent_max_tokens
        )
        self.subagent_reasoning_effort = (
            reasoning_effort if subagent_reasoning_effort is None else subagent_reasoning_effort
        )
        self.subagent_stream = stream if subagent_stream is None else subagent_stream

        self.registry = registry or Registry()
        self.calibrator = TokenCalibrator()
        self.plan_mode = PlanMode(project_dir)
        self.tool_ctx = ToolContext(self)
        self._tool_diffs: dict[str, str] = {}
        self._tool_diffs_lock = threading.Lock()
        # thread-local: sub-agents each execute tools in their own
        # background thread; the "currently executing call" that
        # Edit/Write attach their diff to must be per-thread, or
        # concurrent sub-agents would clobber each other's diff slot
        self._active_call = threading.local()
        # serializes interactive prompts (Question tool, PlanExit
        # confirmation): the TUI can only ask one question at a time
        self._interactive_lock = threading.Lock()
        # dedicated per-invocation sub-agent clients (see run_subagent):
        # concurrent sub-agents each run on their own Client clone, so
        # one sub-agent's connection failure / abort can never tear
        # down a sibling's in-flight request on a shared client.  The
        # active clones are tracked so cancel()/close() can reach them.
        self._subagent_clients_lock = threading.Lock()
        self._active_subagent_clients: list[Client] = []
        self.store = SessionStore(
            project_dir=project_dir,
            model=model,
            backend=backend,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            tool_names=tool_names or config.DEFAULT_TOOLS,
        )

        self.context_ratio: float | None = None
        self.compacting = False
        self.todos: list[dict] = []
        self.pending_user_prompts: list[str] = []
        self._save_error: str | None = None
        self.last_messages: list = []
        self.cancel_event = threading.Event()
        # Monotonic cancel identity: cancel() bumps this counter, so a
        # worker from a cancelled run can tell it was cancelled even
        # after the next run clears the shared event.
        self.cancel_generation = 0
        # Monotonic run identity: bumped when a new top-level run starts
        # (tui._start_agent).  A worker whose captured value no longer
        # matches is stale — superseded by a newer run — and must never
        # touch shared state.  Unlike cancel_generation this is NOT
        # bumped by cancel(): a cancelled run with no successor still
        # owns the session and may salvage its partial history.
        self.run_generation = 0
        self._skill_dir = self._find_skill_dir()

        # TUI hooks (overridden by the UI)
        self.on_delta: Callable[[str], None] | None = None
        self.log_fn: Callable[[str], None] | None = None
        self.notify_fn: Callable[[str, Any], None] | None = None
        self.confirm_fn: Callable[[str], bool] | None = None
        self.ask_fn: Callable[[list[dict]], str] | None = None

    # ------------------------------------------------------------------
    # notifications
    # ------------------------------------------------------------------
    def notify(self, kind: str, data: Any = None) -> None:
        if self.notify_fn:
            self.notify_fn(kind, data)

    def log(self, msg: str) -> None:
        if self.log_fn:
            self.log_fn(msg)

    def confirm(self, prompt: str) -> bool:
        with self._interactive_lock:
            if self.confirm_fn:
                return self.confirm_fn(prompt)
            return True

    def ask_questions(self, questions: list[dict]) -> str:
        with self._interactive_lock:
            if self.ask_fn:
                return self.ask_fn(questions)
            return "Unanswered"

    # ------------------------------------------------------------------
    # tools
    # ------------------------------------------------------------------
    def tool_specs(self, exclude: tuple[str, ...] = ()) -> list:
        """Tool specs exposed to the model; ``exclude`` drops tools by
        name (e.g. one-shot/interactive tools for sub-agent runs)."""
        return [spec for spec in self.registry.specs() if spec.name not in exclude]

    def execute_tool(
        self, name: str, args: dict[str, Any], call_id: str | None = None
    ) -> str | PendingToolResult:
        """Execute a tool.

        ``call_id`` (when given) lets Edit/Write attach a unified diff
        for the TUI to render; retrieve it afterwards with
        ``take_diff(call_id)``.
        """
        # plan-mode guard: only the plan file is writable
        if self.plan_mode.is_plan and name in ("Write", "Edit", "Insert", "Mkdir", "Bash"):
            blocked = self._plan_blocked(name, args)
            if blocked:
                return blocked

        self._active_call.call_id = call_id
        try:
            result = self.registry.execute(name, args, self.tool_ctx)
        finally:
            self._active_call.call_id = None

        self.notify("tool")
        return result

    def record_diff(self, diff_text: str) -> None:
        """Attach a unified diff to the tool call currently executing."""
        call_id = getattr(self._active_call, "call_id", None)
        if call_id and diff_text:
            with self._tool_diffs_lock:
                self._tool_diffs[call_id] = diff_text

    def take_diff(self, call_id: str) -> str | None:
        """Pop and return the diff recorded for CALL_ID, if any."""
        with self._tool_diffs_lock:
            return self._tool_diffs.pop(call_id, None)

    def _plan_blocked(self, name: str, args: dict[str, Any]) -> str | None:
        if name == "Bash":
            return (
                "Error: blocked by plan mode (read-only phase); "
                "Bash is disabled — use Read/Glob/Grep for read-only access"
            )
        path = self._tool_path(name, args)
        if path and path != self.plan_mode.plan_file:
            # Edit diff-mode (no old_str, diff not explicitly False) runs
            # `patch` which can write to arbitrary files via relative paths
            # in the diff content — block it even if the target path looks
            # innocent, because patch follows paths within the diff.
            if name == "Edit" and args.get("old_str") is None and args.get("diff") is not False:
                return (
                    "Error: blocked by plan mode (read-only phase); "
                    "diff/patch mode cannot target files other than the plan "
                    "file — use string replacement (old_str/new_str) instead"
                )
            return (
                "Error: blocked by plan mode (read-only phase); only the plan file may be modified"
            )
        return None

    def _tool_path(self, name: str, args: dict[str, Any]) -> str | None:
        if name == "Write":
            return os.path.realpath(
                os.path.join(str(args.get("path", "")), str(args.get("filename", "")))
            )
        if name == "Edit":
            return os.path.realpath(str(args.get("path", "")))
        if name == "Insert":
            return os.path.realpath(str(args.get("path", "")))
        if name == "Mkdir":
            return os.path.realpath(
                os.path.join(str(args.get("parent", "")), str(args.get("name", "")))
            )
        return None

    # ------------------------------------------------------------------
    # ToolContext-facing API
    # ------------------------------------------------------------------
    def update_todos(self, todos: list[dict]) -> None:
        """Store TODOS so the pinned TUI panel shows the current list."""
        self.todos = list(todos)
        self.notify("todos")

    def clear_todos(self) -> None:
        """Drop the todo list (e.g. session cleared or restored)."""
        self.todos = []
        self.notify("todos")

    def find_skill(self, name: str) -> str | None:
        if not self._skill_dir:
            return None
        skill_root = os.path.realpath(self._skill_dir)
        # Check subdirectory with SKILL.md (e.g. skills/cba-rules/SKILL.md)
        p = os.path.realpath(os.path.join(skill_root, name, "SKILL.md"))
        if p.startswith(skill_root + os.sep) and os.path.isfile(p):
            return p
        # Fallback: flat file (e.g. skills/cba-rules.md or .txt)
        for ext in (".md", ".txt"):
            p = os.path.realpath(os.path.join(skill_root, name + ext))
            if p.startswith(skill_root + os.sep) and os.path.isfile(p):
                return p
        return None

    def _find_skill_dir(self) -> str | None:
        return find_skill_dir(self.project_dir, self._configured_skill_path)

    def run_subagent(self, subagent_type: str, description: str, prompt: str) -> str:
        """Run a delegated sub-agent task.

        The sub-agent has no TodoWrite (parent-only), so it can never
        touch the parent's todo list.

        Each invocation runs on a DEDICATED client, cloned from the
        configured sub-agent client: concurrent Agent tool calls share
        this session, and a shared Client would race — ``_reset_http``
        / ``abort`` swap and close the underlying httpx pool and
        ``_aborted`` is per-request state, so one sub-agent's
        connection failure (or a Ctrl-C) would tear down a sibling's
        in-flight request.  The clone is tracked for cancel/close and
        released when the sub-agent finishes.
        """
        client, owned = self._new_subagent_client()
        if owned:
            with self._subagent_clients_lock:
                self._active_subagent_clients.append(client)
        try:
            return run_subagent(self, description, prompt, client=client)
        finally:
            if owned:
                with self._subagent_clients_lock:
                    if client in self._active_subagent_clients:
                        self._active_subagent_clients.remove(client)
                client.close()

    def _new_subagent_client(self) -> tuple[Any, bool]:
        """A dedicated Client for one sub-agent invocation.

        Real Clients are cloned (fresh httpx pool, own ``_aborted``
        flag, same endpoint/credentials/log).  A non-Client
        ``subagent_client`` (a test double) is passed through
        untouched — the isolation concern does not apply to it, and
        custom clients keep working as-is.
        """
        base = self.subagent_client
        if isinstance(base, Client):
            return base.clone(), True
        return base, False

    def plan_exit(self) -> str:
        """PlanExit tool implementation.

        Asks the user to approve the plan→build switch through a y/n
        confirmation UI (rendered like the Question tool's choice list,
        but keyed with y/n instead of numbers).  The TUI hook decides
        the exact look; the session only interprets the boolean answer.
        """
        if not self.plan_mode.is_plan:
            return "Not in plan mode; PlanExit has no effect.  Continue as normal."
        approved = self.confirm(
            f"Plan at {self.plan_mode.plan_file} is complete. "
            "Switch to build agent and start implementing?"
        )
        if approved:
            self.switch_to_build()
            msg = config.PLAN_EXIT_APPROVED_MESSAGE % (self.plan_mode.plan_file or "")
            self.pending_user_prompts.append(msg)
            return (
                "User approved switching to build agent.  You are now in "
                "build mode; proceed to execute the approved plan."
            )
        return (
            "User rejected switching to build.  Remain in plan mode: keep "
            "planning and refining the plan file, and do NOT edit any other files."
        )

    # ------------------------------------------------------------------
    # mode switching
    # ------------------------------------------------------------------
    def switch_to_build(self) -> None:
        self.plan_mode.set_mode(AgentMode.BUILD, self._mode_prompts())
        self.registry.unregister("PlanExit")

    def switch_to_plan(self) -> None:
        from .tools import PlanExit

        self.plan_mode.set_mode(AgentMode.PLAN, self._mode_prompts())
        self.registry.register(PlanExit())

    def _mode_prompts(self) -> dict[str, str]:
        from .prompts import read_prompt_file

        return {
            "plan": read_prompt_file("plan.md"),
            "plan-mode": read_prompt_file("plan-mode.md"),
            "build-switch": read_prompt_file("build-switch.md"),
        }

    # ------------------------------------------------------------------
    # session persistence hooks
    # ------------------------------------------------------------------
    def remember_user_text(self, messages: list) -> None:
        """Remember the last real user message for session-title generation.

        Skips harness-injected messages (nudges, plan/build-switch
        reminders, queued mode prompts — flagged ``injected``) so a
        title is never generated from "Review the original user request
        and the Task Completion Rules…" or a mode-switch reminder.
        """
        nudge = config.NUDGE_MESSAGE
        for m in reversed(messages):
            if m.role == "user" and not m.injected and m.text() != nudge:
                self.store.remember_first_user_message(m.text())
                break

    def auto_save(self, messages: list, system: str | None) -> None:
        if not config.AUTO_SAVE_SESSION:
            return
        text = self._conversation_text(messages)
        # retry once: transient failures (NFS hiccup, brief lock) clear
        # on the second attempt; permanent ones (disk full, read-only)
        # fail again and leave a persistent, visible error state instead
        # of silently dropping the session
        for attempt in (1, 2):
            try:
                self.store.save(text)
                self._save_error = None
                return
            except OSError as e:
                self._save_error = str(e)
                if attempt == 1:
                    time.sleep(0.2)
        self.log(f"auto-save failed: {self._save_error}")
        self.notify("save-error")

    def generate_session_title(self) -> None:
        """Generate a title from the first real user message (title.md).

        Mirrors gptel-agent-harness--generate-session-title: one-shot per
        session (guarded by store.title / title_pending); on success the
        session file is renamed to <title>_<TS>.md.

        Reasoning models answer with a reasoning preamble; the client
        merges it ahead of the real answer, so it is stripped here or
        the first 50 chars of the reasoning would become the session
        name.  The session temperature is passed so the title request
        matches the buffer settings (elisp parity) instead of the API
        default.
        """
        store = self.store
        if store.title or store.title_pending:
            return
        first = store.first_user_message()
        if not first:
            return
        store.title_pending = True
        try:
            from .models import Message as Msg
            from .prompts import read_prompt_file

            system = read_prompt_file("title.md")
            resp, _ = self.client.chat_sync(
                [Msg(role="user", content=first)],
                system=system,
                temperature=self.temperature,
            )
            title = resp.text_without_reasoning()
            if title:
                store.apply_title(title)
                if self.store.title:
                    self.log(f"session titled — {self.store.title}")
        except Exception as e:  # noqa: BLE001 - title failure is non-fatal
            self.log(f"title generation failed: {e}")
        finally:
            store.title_pending = False

    def _conversation_text(self, messages: list) -> str:
        parts: list[str] = []
        for m in messages:
            body = m.text()
            if m.role == "assistant" and m.tool_calls:
                calls = ", ".join(tc.name for tc in m.tool_calls)
                body = (body + f"\n[tool calls: {calls}]").strip()
            if body:
                parts.append(f"**{m.role}**: {body}")
        return "\n\n".join(parts)

    def close(self) -> None:
        self.cancel()
        self.alive = False
        self.plan_mode.cleanup_plan_file()
        cleanup_spooled_files()
        if hasattr(self.client, "close"):
            self.client.close()
        if self.subagent_client is not self.client and hasattr(self.subagent_client, "close"):
            self.subagent_client.close()
        # defensive: sub-agent workers close their own clones in
        # run_subagent's finally; close any stragglers (e.g. a worker
        # still winding down after cancel) so no pool leaks
        with self._subagent_clients_lock:
            strays = list(self._active_subagent_clients)
            self._active_subagent_clients.clear()
        for c in strays:
            if hasattr(c, "close"):
                with contextlib.suppress(Exception):  # best effort
                    c.close()

    def cancel(self) -> None:
        """Cancel the in-flight agent run (Ctrl-C).

        Sets the cancel event (checked by the agent loop) and aborts the
        active HTTP stream so a blocking read unblocks immediately.  The
        loop turns this into a clean stop, not an error.

        The generation counter makes the cancellation stick to the run
        that was active: a stale worker finishing late (e.g. after a
        long tool call) stays cancelled even once the next run clears
        the shared event, so it can never clobber the new run's state.
        """
        self.cancel_event.set()
        self.cancel_generation += 1
        # A sub-agent streams on its own client when a separate LLM is
        # configured — abort BOTH pools so a blocked sub-agent read is
        # interrupted too (see Client.abort for why close() alone is
        # not enough).  A shared client is aborted once; dedicated
        # per-invocation sub-agent clones (see run_subagent) are each
        # aborted so every in-flight sub-agent request is interrupted.
        clients = [self.client]
        if self.subagent_client is not self.client:
            clients.append(self.subagent_client)
        with self._subagent_clients_lock:
            clients.extend(self._active_subagent_clients)
        for c in clients:
            if hasattr(c, "abort"):
                with contextlib.suppress(Exception):  # best effort
                    c.abort()

    # ------------------------------------------------------------------
    # direct commands: compact / summary (no agent loop)
    # ------------------------------------------------------------------
    def compact_conversation(self) -> tuple[bool, str]:
        """Compact the current conversation in place.

        Mirrors gptel-agent-harness-commands-compact-buffer: the whole
        conversation is sent as the user message with the compact prompt
        as system (tools/stream disabled); on success the conversation is
        replaced by the summary frame only.  Re-appending the last user
        request is the automatic path's job (``AgentLoop.compact``),
        which resumes the run with it; the manual command just replaces
        the history and waits for the next user message.
        """
        from .models import Message as Msg
        from .prompts import read_prompt_file

        # Replacing the conversation is a new generation: invalidate any
        # worker still winding down from a cancelled run, or its
        # salvaged-history commit would clobber the compacted buffer.
        self.run_generation += 1
        messages = self.last_messages or []
        if not messages:
            return False, "Nothing to compact."
        if self.compacting:
            return False, "Compaction already in progress."
        self.compacting = True
        try:
            conversation = self._conversation_text(messages)
            system = read_prompt_file("compact.md")
            resp, _ = self.client.chat_sync([Msg(role="user", content=conversation)], system=system)
            summary = resp.text_without_reasoning()
            if not summary:
                return False, "Compaction failed: empty summary."
            frame = config.COMPACT_HEADER + summary + config.COMPACT_SEPARATOR
            # The summary is part of the user turn (the original system
            # prompt is passed separately and stays untouched), so it
            # replaces the history as a user message — matching the
            # elisp flow where the compacted summary is plain buffer
            # text sent as the user prompt.
            self.last_messages = [
                Msg(role="user", content=frame.strip()),
            ]
            self.auto_save(self.last_messages, self.system_prompt)
            self.notify("compact")
            return True, "Buffer compacted successfully."
        except Exception as e:  # noqa: BLE001 - compaction failure is non-fatal
            self.log(f"compaction failed: {e}")
            return False, f"Compaction failed: {e}"
        finally:
            self.compacting = False

    def summarize_conversation(self) -> str:
        """Append a summary of the conversation (tools disabled).

        Mirrors gptel-agent-harness-commands-summary: the conversation
        text is sent with the summary prompt, and the result is appended
        as an assistant message plus a session save.
        """
        from .models import Message as Msg
        from .prompts import read_prompt_file

        # Appending to the shared conversation is a new generation: invalidate
        # any worker still winding down from a cancelled run, or its
        # salvaged-history commit would clobber the appended summary.
        self.run_generation += 1
        messages = self.last_messages or []
        if not messages:
            return "Nothing to summarize."
        conversation = self._conversation_text(messages)
        system = read_prompt_file("summary.md")
        try:
            resp, _ = self.client.chat_sync([Msg(role="user", content=conversation)], system=system)
            summary = resp.text_without_reasoning()
        except Exception as e:  # noqa: BLE001
            return f"Summary failed: {e}"
        if not summary:
            return "Summary failed: empty response."
        self.last_messages.append(Msg(role="assistant", content=summary))
        self.auto_save(self.last_messages, self.system_prompt)
        return "Summary appended."
