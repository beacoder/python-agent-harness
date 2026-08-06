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
from .session import SessionStore
from .subagent import run_subagent
from .tokenizer import TokenCalibrator
from .tools import Registry, ToolContext
from .undo import UndoStack


def find_skill_dir(project_dir: str) -> str | None:
    """Locate the skill directory (first match wins)."""
    for cand in (
        os.path.join(os.path.expanduser("~"), ".emacs.d", "skills"),
        os.path.join(project_dir, "skills"),
    ):
        if os.path.isdir(cand):
            return cand
    return None


def find_context_dir(project_dir: str) -> str | None:
    """Locate the default context directory (first match wins)."""
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
        self.pending_user_prompts: list[str] = []
        self._pending_execute_prompt: str | None = None
        self.last_messages: list = []
        self.cancel_event = threading.Event()
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
    def tool_specs(self) -> list:
        return self.registry.specs()

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
        self.todos = todos
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
        return find_skill_dir(self.project_dir)
        return None

    def run_subagent(self, subagent_type: str, description: str, prompt: str) -> str:
        return run_subagent(self, description, prompt)

    def plan_exit(self) -> str:
        """PlanExit tool implementation."""
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
        from .compaction import read_prompt_file

        return {
            "plan": read_prompt_file("plan.txt"),
            "plan-mode": read_prompt_file("plan-mode.txt"),
            "build-switch": read_prompt_file("build-switch.txt"),
        }

    # ------------------------------------------------------------------
    # session persistence hooks
    # ------------------------------------------------------------------
    def remember_user_text(self, messages: list) -> None:
        for m in reversed(messages):
            if m.role == "user":
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
        """
        self.cancel_event.set()
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
        from .compaction import last_user_request, read_prompt_file
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
        from .compaction import read_prompt_file
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
