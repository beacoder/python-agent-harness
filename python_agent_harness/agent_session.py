"""AgentSession: the runtime hub wiring tools, safety, cache, plan mode.

The session implements the ToolContext-facing API (path guards, bash
verdicts, snapshots, cache invalidation, sub-agents, questions) and the
agent-loop-facing API (client, calibrator, plan mode, auto-save,
notifications).  The TUI layer subclasses it to provide interactive
confirmations.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Callable

from . import config
from .cache import ToolCache
from .client import Client
from .models import AgentMode
from .planmode import PlanMode
from .safety import BashPolicy, SafetyViolation, check_path
from .session_store import SessionStore
from .subagent import run_subagent
from .token_estimator import TokenCalibrator
from .tools import Registry, ToolContext
from .undo import UndoStack


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
        max_tokens: int = config.MAX_TOKENS,
        reasoning_effort: str | None = None,
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
        self.tools_enabled = True
        self.alive = True
        self._configured_context_path = context_path
        self._configured_skill_path = skill_path

        self.registry = registry or Registry()
        self.cache = ToolCache()
        self.calibrator = TokenCalibrator()
        self.plan_mode = PlanMode(project_dir)
        self.undo = UndoStack()
        self.bash_policy = BashPolicy()
        self.tool_ctx = ToolContext(self)
        self._tool_diffs: dict[str, str] = {}
        self._active_call_id: str | None = None
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
        # scoped todo lists: "main" is the top-level agent; each running
        # sub-agent gets its own scope so its TodoWrite calls don't
        # clobber the parent's list
        self._todo_scopes: dict[str, list[dict]] = {"main": []}
        self._todo_scope_stack: list[str] = ["main"]
        self._todo_scope_labels: dict[str, str] = {}
        self._subagent_seq = 0
        self.pending_user_prompts: list[str] = []
        self._pending_execute_prompt: str | None = None
        self.last_messages: list = []
        self.cancel_event = threading.Event()
        # Monotonic cancel identity: cancel() bumps this counter, so a
        # worker from a cancelled run can tell it was cancelled even
        # after the next run clears the shared event.
        self.cancel_generation = 0
        self._skill_dir = self._find_skill_dir()

        # TUI hooks (overridden by the UI)
        self.on_delta: Callable[[str], None] | None = None
        self.log_fn: Callable[[str], None] | None = None
        self.notify_fn: Callable[[str], None] | None = None
        self.confirm_fn: Callable[[str], bool] | None = None
        self.ask_fn: Callable[[list[dict]], str] | None = None
        self.bash_approval_fn: Callable[[str], tuple[bool, str]] | None = None

    # ------------------------------------------------------------------
    # notifications
    # ------------------------------------------------------------------
    def notify(self, kind: str) -> None:
        if self.notify_fn:
            self.notify_fn(kind)

    def log(self, msg: str) -> None:
        if self.log_fn:
            self.log_fn(msg)

    def confirm(self, prompt: str) -> bool:
        if self.confirm_fn:
            return self.confirm_fn(prompt)
        return True

    def ask_questions(self, questions: list[dict]) -> str:
        if self.ask_fn:
            return self.ask_fn(questions)
        return "Unanswered"

    # ------------------------------------------------------------------
    # tools
    # ------------------------------------------------------------------
    def tool_specs(self, exclude: tuple[str, ...] = ()) -> list:
        """Tool specs exposed to the model; ``exclude`` drops tools by
        name (e.g. one-shot/interactive tools for sub-agent runs)."""
        return [
            spec for spec in self.registry.specs()
            if spec.name not in exclude
        ]

    def execute_tool(
        self, name: str, args: dict[str, Any], call_id: str | None = None
    ) -> str:
        """Execute a tool with cache + safety integration.

        ``call_id`` (when given) lets Edit/Write attach a unified diff
        for the TUI to render; retrieve it afterwards with
        ``take_diff(call_id)``.
        """
        # cache path for read/glob/grep
        cached = self._cache_get(name, args)
        if cached is not None:
            return cached
        if name in ("Read", "Glob", "Grep"):
            self.cache.misses += 1

        # plan-mode guard: only the plan file is writable
        if self.plan_mode.is_plan and name in ("Write", "Edit", "Insert", "Mkdir", "Bash"):
            blocked = self._plan_blocked(name, args)
            if blocked:
                return blocked

        if name in ("Write", "Edit", "Insert", "Mkdir"):
            path = self._tool_path(name, args)
            if path:
                try:
                    check_path(path, name)
                except SafetyViolation as e:
                    return str(e)

        self._active_call_id = call_id
        try:
            result = self.registry.execute(name, args, self.tool_ctx)
        finally:
            self._active_call_id = None

        # cache store + write-through invalidation
        if self.cache.cacheable_p(result):
            self._cache_store(name, args, result)
        if name in ("Write", "Edit", "Insert"):
            path = self._tool_path(name, args)
            if path:
                self.cache.invalidate_path(path)
        self.notify("tool")
        return result

    def record_diff(self, diff_text: str) -> None:
        """Attach a unified diff to the tool call currently executing."""
        if self._active_call_id and diff_text:
            self._tool_diffs[self._active_call_id] = diff_text

    def take_diff(self, call_id: str) -> str | None:
        """Pop and return the diff recorded for CALL_ID, if any."""
        return self._tool_diffs.pop(call_id, None)

    def _plan_blocked(self, name: str, args: dict[str, Any]) -> str | None:
        if name == "Bash":
            return None  # handled by bash policy below
        path = self._tool_path(name, args)
        if path and path != self.plan_mode.plan_file:
            return (
                "Error: blocked by plan mode (read-only phase); "
                "only the plan file may be modified"
            )
        return None

    def _tool_path(self, name: str, args: dict[str, Any]) -> str | None:
        if name == "Write":
            return os.path.abspath(os.path.join(str(args.get("path", "")), str(args.get("filename", ""))))
        if name == "Edit":
            return os.path.abspath(str(args.get("path", "")))
        if name == "Insert":
            return os.path.abspath(str(args.get("path", "")))
        if name == "Mkdir":
            return os.path.abspath(os.path.join(str(args.get("parent", "")), str(args.get("name", ""))))
        return None

    # ------------------------------------------------------------------
    # cache integration
    # ------------------------------------------------------------------
    def _cache_get(self, name: str, args: dict[str, Any]) -> str | None:
        key, path = self._cache_key(name, args)
        if key is None:
            return None
        return self.cache.get(key, path)

    def _cache_store(self, name: str, args: dict[str, Any], result: str) -> None:
        key, path = self._cache_key(name, args)
        if key is None:
            return
        self.cache.store(key, result, path)
        self.cache.mark_seen(key)

    def _cache_key(self, name: str, args: dict[str, Any]) -> tuple[tuple | None, str | None]:
        if name == "Read":
            path = os.path.abspath(str(args.get("file_path", "")))
            return (("read", path, args.get("start_line"), args.get("end_line")), path)
        if name == "Glob":
            base = os.path.abspath(str(args.get("path") or self.project_dir))
            return (("glob", str(args.get("pattern")), base, args.get("depth")), base)
        if name == "Grep":
            path = os.path.abspath(str(args.get("path", "")))
            return (("grep", str(args.get("regex")), path, args.get("glob"), args.get("context_lines")), path)
        return None, None

    # ------------------------------------------------------------------
    # ToolContext-facing API
    # ------------------------------------------------------------------
    def guard_path(self, path: str, tool_name: str) -> None:
        check_path(path, tool_name)

    def verify_bash(self, command: str) -> str | None:
        """Return an error string to deliver, or None to run."""
        self.bash_policy.plan_mode = self.plan_mode.is_plan
        verdict = self.bash_policy.verdict(command)
        if verdict != "CONFIRM":
            return verdict
        if self.bash_approval_fn:
            run, answer = self.bash_approval_fn(command)
        else:
            run, answer = self._ask_via_tui(command)
        if answer == "allow":
            self.bash_policy.record(command, "allow")
            return None
        if answer == "deny":
            self.bash_policy.record(command, "deny")
            return "Error: Bash command rejected by user approval (denied for this session)."
        if run:
            return None
        return "Error: Bash command rejected by user approval."

    def _ask_via_tui(self, command: str) -> tuple[bool, str]:
        prompt = (
            "Dangerous Bash command:\n\n"
            f"{command}\n\n"
            "Run it? [y]es / [n]o / [a]lways allow (session) / [d]eny (session)"
        )
        if self.ask_fn is None:
            # headless fallback: run once (matches confirm-tool-calls opt-out)
            return True, "run"
        answer = self.ask_fn([{"question": prompt}])
        if answer.startswith("a"):
            return True, "allow"
        if answer.startswith("d"):
            return False, "deny"
        if answer.startswith("y") or "Yes" in answer:
            return True, "run"
        return False, "run"

    def snapshot(self, path: str, tool: str) -> None:
        self.undo.snapshot(path, tool)

    def record_absent(self, path: str, tool: str) -> None:
        self.undo.record_absent(path, tool)

    def invalidate_cache(self, path: str) -> None:
        self.cache.invalidate_path(path)

    def update_todos(self, todos: list[dict]) -> None:
        """Store TODOS into the currently active scope.

        While a sub-agent runs, its TodoWrite calls land in the
        sub-agent's own scope; `self.todos` always mirrors the active
        scope so the pinned TUI panel shows the right list.
        """
        scope = self._todo_scope_stack[-1]
        self._todo_scopes[scope] = list(todos)
        self.todos = list(todos)
        self.notify("todos")

    @property
    def todo_scope_label(self) -> str | None:
        """Human-readable label of the active scope (None for main)."""
        scope = self._todo_scope_stack[-1]
        return self._todo_scope_labels.get(scope)

    def push_todo_scope(self, scope_id: str, label: str | None = None) -> None:
        """Switch the active todos scope (e.g. when a sub-agent starts)."""
        if scope_id not in self._todo_scopes:
            self._todo_scopes[scope_id] = []
        if label:
            self._todo_scope_labels[scope_id] = label
        self._todo_scope_stack.append(scope_id)
        self.todos = list(self._todo_scopes[scope_id])
        self.notify("todos")

    def pop_todo_scope(self) -> None:
        """Restore the previous todos scope (e.g. sub-agent finished)."""
        if len(self._todo_scope_stack) > 1:
            self._todo_scope_stack.pop()
            self.todos = list(self._todo_scopes[self._todo_scope_stack[-1]])
            self.notify("todos")

    def clear_todos(self) -> None:
        """Drop all todo scopes (e.g. session cleared or restored)."""
        self._todo_scopes = {"main": []}
        self._todo_scope_stack = ["main"]
        self._todo_scope_labels = {}
        self.todos = []
        self.notify("todos")

    def find_skill(self, name: str) -> str | None:
        if not self._skill_dir:
            return None
        # Check subdirectory with SKILL.md (e.g. skills/cba-rules/SKILL.md)
        p = os.path.join(self._skill_dir, name, "SKILL.md")
        if os.path.isfile(p):
            return p
        # Fallback: flat file (e.g. skills/cba-rules.md or .txt)
        for ext in (".md", ".txt"):
            p = os.path.join(self._skill_dir, name + ext)
            if os.path.isfile(p):
                return p
        return None

    def _find_skill_dir(self) -> str | None:
        return find_skill_dir(self.project_dir, self._configured_skill_path)

    def run_subagent(self, subagent_type: str, description: str, prompt: str) -> str:
        """Run a delegated sub-agent task with an isolated todos scope.

        The sub-agent's TodoWrite calls go to its own scope (visible in
        the pinned TUI panel with a `sub:` label); when it finishes the
        parent's todo list is restored automatically.
        """
        self._subagent_seq += 1
        scope_id = f"sub:{self._subagent_seq}"
        self.push_todo_scope(scope_id, description)
        try:
            return run_subagent(self, description, prompt)
        finally:
            self.pop_todo_scope()

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
            msg = config.PLAN_EXIT_APPROVED_MESSAGE % (
                self.plan_mode.plan_file or ""
            )
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
            "plan": read_prompt_file("plan.txt"),
            "plan-mode": read_prompt_file("plan-mode.txt"),
            "build-switch": read_prompt_file("build-switch.txt"),
        }

    # ------------------------------------------------------------------
    # session persistence hooks
    # ------------------------------------------------------------------
    def remember_user_text(self, messages: list) -> None:
        """Remember the last real user message for session-title generation.

        Skips harness-injected nudges so a title is never generated from
        "Review the original user request and the Task Completion Rules…".
        """
        nudge = config.NUDGE_MESSAGE
        for m in reversed(messages):
            if m.role == "user" and m.text() != nudge:
                self.store.remember_first_user_message(m.text())
                break

    def auto_save(self, messages: list, system: str | None) -> None:
        if not config.AUTO_SAVE_SESSION:
            return
        text = self._conversation_text(messages)
        try:
            self.store.save(text)
        except OSError as e:
            self.log(f"auto-save failed: {e}")

    def generate_session_title(self) -> None:
        """Generate a title from the first real user message (title.txt).

        Mirrors gptel-agent-harness--generate-session-title: one-shot per
        session (guarded by store.title / title_pending); on success the
        session file is renamed to <title>_<TS>.md.
        """
        store = self.store
        if store.title or store.title_pending:
            return
        first = store.first_user_message()
        if not first:
            return
        store.title_pending = True
        try:
            from .prompts import read_prompt_file
            from .models import Message as Msg

            system = read_prompt_file("title.txt")
            resp, _ = self.client.chat_sync(
                [Msg(role="user", content=first)], system=system
            )
            title = resp.text()
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
        if hasattr(self.client, "close"):
            self.client.close()

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
        if hasattr(self.client, "abort"):
            try:
                self.client.abort()
            except Exception:  # noqa: BLE001 - best effort
                pass

    # ------------------------------------------------------------------
    # direct commands: compact / summary (no agent loop)
    # ------------------------------------------------------------------
    def compact_conversation(self) -> tuple[bool, str]:
        """Compact the current conversation in place.

        Mirrors gptel-agent-harness-commands-compact-buffer: the whole
        conversation is sent as the user message with the compact prompt
        as system (tools/stream disabled); on success the conversation is
        replaced by the summary frame + the last real user request, the
        cache epoch is reset, and the session file is refreshed.
        """
        from .prompts import last_user_request, read_prompt_file
        from .models import Message as Msg

        messages = self.last_messages or []
        if not messages:
            return False, "Nothing to compact."
        if self.compacting:
            return False, "Compaction already in progress."
        request = last_user_request([m.to_api() for m in messages])
        if not request:
            return False, "No user request to resume after compaction."
        self.compacting = True
        try:
            conversation = self._conversation_text(messages)
            system = read_prompt_file("compact.txt")
            resp, _ = self.client.chat_sync(
                [Msg(role="user", content=conversation)], system=system
            )
            summary = resp.text()
            if not summary:
                return False, "Compaction failed: empty summary."
            self.cache.reset_epoch()
            frame = config.COMPACT_HEADER + summary + config.COMPACT_SEPARATOR
            self.last_messages = [
                Msg(role="system", content=frame.strip()),
                Msg(role="user", content=request),
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
        from .prompts import read_prompt_file
        from .models import Message as Msg

        messages = self.last_messages or []
        if not messages:
            return "Nothing to summarize."
        conversation = self._conversation_text(messages)
        system = read_prompt_file("summary.txt")
        try:
            resp, _ = self.client.chat_sync(
                [Msg(role="user", content=conversation)], system=system
            )
            summary = resp.text()
        except Exception as e:  # noqa: BLE001
            return f"Summary failed: {e}"
        if not summary:
            return "Summary failed: empty response."
        self.last_messages.append(Msg(role="assistant", content=summary))
        self.auto_save(self.last_messages, self.system_prompt)
        return "Summary appended."
