"""Tests for the remaining tool modules: base (ToolContext/Registry),
Question, PlanExit, Skill, TodoWrite, Bash internals, and AgentTool.

These tools are normally exercised end-to-end through Session
integration tests; this file drives them directly through their
injection points (session callbacks) so the interactive/async paths and
error containment boundaries are covered.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from python_agent_harness.tools.agent_tool import AgentTool
from python_agent_harness.tools.base import PendingToolResult, Registry, ToolContext
from python_agent_harness.tools.planexit import PlanExit
from python_agent_harness.tools.question import Question
from python_agent_harness.tools.skill import Skill
from python_agent_harness.tools.todo import TodoWrite


class FakeSession:
    """Session double satisfying the ToolContext callback protocol."""

    def __init__(self) -> None:
        self.project_dir = "/tmp"
        self.received_todos: list[dict] = []
        self._cancel = threading.Event()

    def ask_questions(self, questions: list[dict]) -> str:
        return "answer"

    def record_diff(self, diff_text: str) -> None:
        pass

    def update_todos(self, todos: list[dict]) -> None:
        self.received_todos = todos

    def find_skill(self, name: str) -> str | None:
        return None

    def run_subagent(self, subagent_type: str, description: str, prompt: str) -> str:
        return f"ran {description}"

    def plan_exit(self) -> str:
        return "approved"

    @property
    def cancel_event(self) -> threading.Event:
        return self._cancel


class TestToolContext(unittest.TestCase):
    """Proxy-to-session and default no-op behavior of ToolContext."""

    def test_cwd_defaults_dot(self):
        self.assertEqual(ToolContext().cwd, ".")

    def test_cwd_uses_session_project_dir(self):
        self.assertEqual(ToolContext(FakeSession()).cwd, "/tmp")

    def test_ask_questions_proxies_to_session(self):
        sess = FakeSession()
        sess.ask_questions = lambda qs: "picked A"
        self.assertEqual(ToolContext(sess).ask_questions([{"question": "q"}]), "picked A")

    def test_ask_questions_defaults_unanswered(self):
        self.assertEqual(ToolContext().ask_questions([{"question": "q"}]), "Unanswered")

    def test_update_todos_proxies_to_session(self):
        sess = FakeSession()
        todos = [{"content": "a", "status": "pending"}]
        ToolContext(sess).update_todos(todos)
        self.assertEqual(sess.received_todos, todos)

    def test_find_skill_proxies_to_session(self):
        sess = FakeSession()
        sess.find_skill = lambda name: "/skills/x.md"
        self.assertEqual(ToolContext(sess).find_skill("x"), "/skills/x.md")

    def test_find_skill_defaults_none(self):
        self.assertIsNone(ToolContext().find_skill("x"))

    def test_run_subagent_defaults_to_error(self):
        out = ToolContext().run_subagent("subagent", "do it", "p")
        self.assertIn("no session", out)
        self.assertIn("do it", out)

    def test_plan_exit_proxies_to_session(self):
        sess = FakeSession()
        sess.plan_exit = lambda: "switched"
        self.assertEqual(ToolContext(sess).plan_exit(), "switched")

    def test_plan_exit_defaults_not_in_plan_mode(self):
        self.assertIn("Not in plan mode", ToolContext().plan_exit())

    def test_cancel_event_proxies_to_session(self):
        sess = FakeSession()
        self.assertIs(ToolContext(sess).cancel_event, sess._cancel)

    def test_cancel_event_defaults_none(self):
        self.assertIsNone(ToolContext().cancel_event)


class TestRegistry(unittest.TestCase):
    """Registry lifecycle and error wrapping."""

    def test_register_get_unregister(self):
        reg = Registry()
        tool = TodoWrite()
        reg.register(tool)
        self.assertIs(reg.get("TodoWrite"), tool)
        reg.unregister("TodoWrite")
        self.assertIsNone(reg.get("TodoWrite"))
        reg.unregister("TodoWrite")  # idempotent: must not raise

    def test_specs_names_filter(self):
        reg = Registry()
        reg.register(TodoWrite())
        reg.register(AgentTool())
        specs = reg.specs(["TodoWrite"])
        self.assertEqual([s.name for s in specs], ["TodoWrite"])

    def test_execute_unknown_tool(self):
        out = Registry().execute("Nope", {}, ToolContext())
        self.assertEqual(out, "Error: unknown tool 'Nope'")

    def test_execute_wraps_tool_exception(self):
        class Boom(TodoWrite):
            def run(self, args, ctx):
                raise RuntimeError("boom")

        reg = Registry()
        reg.register(Boom())
        out = reg.execute("TodoWrite", {}, ToolContext())
        self.assertIn("Error: tool TodoWrite failed", out)


class TestQuestionTool(unittest.TestCase):
    """Question sync contract (mirrors gptel's non-``:async t``
    Question tool): run() blocks and returns the answers as a plain
    string — it executes one at a time, in call order."""

    def test_list_questions_delivers_answer(self):
        sess = FakeSession()
        sess.ask_questions = lambda qs: "42"
        result = Question().run({"questions": [{"question": "q1"}]}, ToolContext(sess))
        self.assertEqual(result, "42")

    def test_dict_wrapped_questions_without_session(self):
        result = Question().run({"questions": {"questions": [{"question": "q"}]}}, ToolContext())
        self.assertEqual(result, "Unanswered")

    def test_invalid_questions_returns_error(self):
        result = Question().run({"questions": "nope"}, ToolContext())
        self.assertEqual(result, "Error: questions must be an array")

    def test_ask_questions_exception_contained(self):
        def boom(qs):
            raise RuntimeError("ui broke")

        sess = FakeSession()
        sess.ask_questions = boom
        result = Question().run({"questions": [{"question": "q"}]}, ToolContext(sess))
        self.assertIn("Error: Question failed", result)


class TestPlanExitTool(unittest.TestCase):
    """PlanExit sync contract (mirrors gptel's non-``:async t``
    PlanExit tool): run() blocks and returns the outcome as a plain
    string; failures contained as error strings."""

    def test_plan_exit_success(self):
        sess = FakeSession()
        sess.plan_exit = lambda: "approved"
        result = PlanExit().run({}, ToolContext(sess))
        self.assertEqual(result, "approved")

    def test_plan_exit_without_session(self):
        result = PlanExit().run({}, ToolContext())
        self.assertEqual(
            result,
            "Not in plan mode; PlanExit has no effect.  Continue as normal.",
        )

    def test_plan_exit_exception_contained(self):
        def boom():
            raise RuntimeError("no")

        sess = FakeSession()
        sess.plan_exit = boom
        result = PlanExit().run({}, ToolContext(sess))
        self.assertIn("Error: PlanExit failed", result)


class TestSkillTool(unittest.TestCase):
    """Skill load: not-found error and content loading from disk."""

    def test_skill_not_found(self):
        out = Skill().run({"skill": "nope"}, ToolContext())
        self.assertIn("skill 'nope' not found", out)

    def test_skill_found_reads_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "skill.md")
            with open(p, "w") as f:
                f.write("instructions")
            sess = FakeSession()
            sess.find_skill = lambda name: p
            out = Skill().run({"skill": "my-skill"}, ToolContext(sess))
            self.assertEqual(out, "[Skill: my-skill]\n\ninstructions")


class TestTodoWriteTool(unittest.TestCase):
    """TodoWrite updates the session and returns the JSON payload."""

    def test_updates_todos_and_returns_json(self):
        sess = FakeSession()
        todos = [
            {"content": "a", "status": "pending"},
            {"content": "b", "status": "in_progress"},
        ]
        out = TodoWrite().run({"todos": todos}, ToolContext(sess))
        self.assertEqual(sess.received_todos, todos)
        parsed = json.loads(out)
        self.assertEqual(parsed["count"], 2)
        self.assertEqual(parsed["todos"], todos)

    def test_missing_todos_defaults_empty(self):
        out = TodoWrite().run({}, ToolContext())
        parsed = json.loads(out)
        self.assertEqual(parsed["todos"], [])
        self.assertEqual(parsed["count"], 0)


class TestAgentTool(unittest.TestCase):
    """Agent async contract: sub-agent result delivered from a
    background thread; failures contained as error strings."""

    def test_empty_prompt_rejected(self):
        out = AgentTool().run({"prompt": ""}, ToolContext())
        self.assertEqual(out, "Error: prompt must not be empty")

    def test_run_subagent_delivers_result(self):
        sess = FakeSession()
        sess.run_subagent = lambda t, d, p: f"result for {d}"
        result = AgentTool().run(
            {"subagent_type": "subagent", "description": "do it", "prompt": "work"},
            ToolContext(sess),
        )
        self.assertIsInstance(result, PendingToolResult)
        self.assertEqual(result.wait(), "result for do it")

    def test_run_subagent_without_session(self):
        result = AgentTool().run({"prompt": "p"}, ToolContext())
        self.assertIn("unexpected response — no session", result.wait())

    def test_agent_tool_contained_subagent_exception(self):
        def boom(t, d, p):
            raise RuntimeError("subagent crashed")

        sess = FakeSession()
        sess.run_subagent = boom
        result = AgentTool().run({"description": "task", "prompt": "p"}, ToolContext(sess))
        self.assertIn("Error: Task 'task' failed", result.wait())


class TestBashInternals(unittest.TestCase):
    """Bash process-group kill, bounded output collection, and
    read-failure containment."""

    def test_kill_pgid_swallows_process_lookup_error(self):
        from python_agent_harness.tools.bash import _kill_pgid

        with mock.patch("os.killpg", side_effect=ProcessLookupError):
            _kill_pgid(1234)  # must not raise

    def test_kill_pgid_swallows_kill_failure(self):
        from python_agent_harness.tools.bash import _kill_pgid

        with mock.patch("os.killpg", side_effect=PermissionError):
            _kill_pgid(1234)  # must not raise

    def test_kill_pgid_kills_stored_group_id(self):
        from python_agent_harness.tools.bash import _kill_pgid

        with mock.patch("os.killpg") as killpg:
            _kill_pgid(42)
        killpg.assert_called_once_with(42, 9)  # SIGKILL, no getpgid

    def test_read_failure_delivered_as_error(self):
        from python_agent_harness.tools.bash import Bash

        fake = mock.Mock()
        fake.stdout = mock.Mock()
        fake.stdout.fileno.side_effect = RuntimeError("pipe broke")
        with mock.patch("python_agent_harness.tools.bash.subprocess.Popen", return_value=fake):
            result = Bash().run({"command": "echo hi"}, ToolContext())
        self.assertIsInstance(result, PendingToolResult)
        self.assertIn("Error: Bash failed", result.wait())

    def test_popen_gets_devnull_stdin(self):
        """Bash must not inherit the harness's stdin: commands that read
        stdin would steal keystrokes from the TUI."""
        import subprocess

        from python_agent_harness.tools.bash import Bash

        r_fd, w_fd = os.pipe()
        fake = mock.Mock()
        fake.pid = 4242
        fake.stdout = os.fdopen(r_fd, "rb", buffering=0)
        fake.poll.return_value = None
        fake.wait.return_value = 0
        with mock.patch(
            "python_agent_harness.tools.bash.subprocess.Popen", return_value=fake
        ) as popen:
            result = Bash().run({"command": "echo hi"}, ToolContext())
        os.close(w_fd)  # EOF: the collector breaks out promptly
        result.wait()
        kwargs = popen.call_args.kwargs
        self.assertIs(kwargs["stdin"], subprocess.DEVNULL)
        self.assertIs(kwargs["stdout"], subprocess.PIPE)
        self.assertIs(kwargs["stderr"], subprocess.STDOUT)

    def test_large_output_truncated_and_bounded(self):
        """A >20 KB output is truncated to head + tail and the delivered
        string stays within the cap (the old code could deliver more
        than the input when a single line exceeded the cap)."""
        from python_agent_harness.tools.bash import _MAX_OUTPUT, Bash

        result = Bash().run({"command": "seq 1 10000"}, ToolContext())
        out = result.wait()
        self.assertIn("[truncated", out)
        self.assertLess(len(out), _MAX_OUTPUT + 300)
        self.assertTrue(out.startswith("1\n2\n"))  # head kept
        self.assertTrue(out.endswith("9999\n10000\nExit code: 0"))  # tail kept + exit code

    def test_single_giant_line_truncated(self):
        """A single line far beyond the cap must not blow the delivered
        size (the old code kept the whole line in the tail)."""
        from python_agent_harness.tools.bash import _MAX_OUTPUT, Bash

        result = Bash().run(
            {"command": f"{sys.executable} -c \"print('a' * 100000)\""}, ToolContext()
        )
        out = result.wait()
        self.assertIn("[truncated", out)
        self.assertLess(len(out), _MAX_OUTPUT + 300)
        self.assertTrue(out.startswith("a" * 100))

    def test_detached_child_holding_pipe_does_not_wedge(self):
        """A daemonized child that keeps the stdout pipe open after the
        shell exits must not block delivery: the collector stops reading
        shortly after the process exits instead of waiting for EOF."""
        from python_agent_harness.tools.bash import Bash

        result = Bash().run({"command": "sleep 2 & echo done"}, ToolContext())
        start = time.monotonic()
        out = result.wait()
        elapsed = time.monotonic() - start
        self.assertEqual(out, "done\nExit code: 0")
        self.assertLess(elapsed, 5)

    def test_cancel_preset_skips_spawn(self):
        """If Ctrl-C is already pending, Bash must not spawn a process
        that would be killed moments later — it returns the cancelled
        error synchronously instead."""
        from python_agent_harness.tools.bash import Bash

        sess = FakeSession()
        sess._cancel.set()
        result = Bash().run({"command": "sleep 30"}, ToolContext(sess))
        self.assertIsInstance(result, str)
        self.assertEqual(result, "Error: Bash command cancelled.")

    def test_cancel_actually_kills_process_group(self):
        """Cancel must leave no orphan behind: the deliverer kills the
        process group itself (no watcher race), so the shell and its
        children die and are reaped."""
        from python_agent_harness.tools.bash import Bash

        with tempfile.TemporaryDirectory(prefix="pah-kill-") as d:
            pidfile = os.path.join(d, "pid")
            sess = FakeSession()
            result = Bash().run({"command": f"echo $$ > {pidfile}; sleep 30"}, ToolContext(sess))
            deadline = time.monotonic() + 3
            while not os.path.exists(pidfile) and time.monotonic() < deadline:
                time.sleep(0.05)
            with open(pidfile) as f:
                pgid = int(f.read().strip())
            sess._cancel.set()
            self.assertIn("cancelled", result.wait())
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                try:
                    os.kill(pgid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.05)
            else:
                self.fail("process group survived cancel")

    def test_silence_timeout_kills_and_reports(self):
        """A command silent for the silence budget is killed (SIGTERM ->
        SIGKILL), reported as timed out, and leaves no orphan behind."""
        from python_agent_harness.tools import bash as bash_mod
        from python_agent_harness.tools.bash import Bash

        with tempfile.TemporaryDirectory(prefix="pah-timeout-") as d:
            pidfile = os.path.join(d, "pid")
            sess = FakeSession()
            with mock.patch.object(bash_mod, "BASH_TIMEOUT_SILENCE", 0.5):
                result = Bash().run(
                    {"command": f"echo $$ > {pidfile}; sleep 30"}, ToolContext(sess)
                )
                deadline = time.monotonic() + 3
                while not os.path.exists(pidfile) and time.monotonic() < deadline:
                    time.sleep(0.05)
                with open(pidfile) as f:
                    pgid = int(f.read().strip())
                start = time.monotonic()
                out = result.wait()
            self.assertIn("timed out", out)
            self.assertIn("no output for", out)
            self.assertLess(time.monotonic() - start, 8)
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                try:
                    os.kill(pgid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.05)
            else:
                self.fail("process group survived silence timeout")

    def test_silence_timeout_activity_resets_timer(self):
        """A command that keeps printing must NOT be killed: the silence
        timer resets on every chunk (the long-build scenario)."""
        from python_agent_harness.tools import bash as bash_mod
        from python_agent_harness.tools.bash import Bash

        sess = FakeSession()
        with mock.patch.object(bash_mod, "BASH_TIMEOUT_SILENCE", 0.5):
            result = Bash().run(
                {
                    "command": (
                        f'{sys.executable} -c "import sys,time; '
                        '[print(i, flush=True) or time.sleep(0.05) for i in range(40)]"'
                    )
                },
                ToolContext(sess),
            )
            out = result.wait()
        self.assertNotIn("timed out", out)
        self.assertIn("Exit code: 0", out)
        self.assertIn("39", out)  # full output delivered

    def test_max_timeout_cap_fires_despite_output(self):
        """BASH_TIMEOUT_MAX is an absolute ceiling: it fires even while
        output is flowing."""
        from python_agent_harness.tools import bash as bash_mod
        from python_agent_harness.tools.bash import Bash

        sess = FakeSession()
        with (
            mock.patch.object(bash_mod, "BASH_TIMEOUT_SILENCE", None),
            mock.patch.object(bash_mod, "BASH_TIMEOUT_MAX", 1.0),
        ):
            result = Bash().run(
                {
                    "command": (
                        f'{sys.executable} -c "import sys,time; '
                        '[print(i, flush=True) or time.sleep(0.1) for i in range(300)]"'
                    )
                },
                ToolContext(sess),
            )
            start = time.monotonic()
            out = result.wait()
        self.assertIn("timed out", out)
        self.assertIn("maximum", out)
        self.assertLess(time.monotonic() - start, 8)

    def test_exit_code_reported(self):
        """Normal completion appends the exit code; it survives
        truncation (always the last line of the kept tail)."""
        from python_agent_harness.tools.bash import Bash

        self.assertTrue(
            Bash().run({"command": "true"}, ToolContext()).wait().endswith("Exit code: 0")
        )
        self.assertTrue(
            Bash().run({"command": "exit 3"}, ToolContext()).wait().endswith("Exit code: 3")
        )
        out = Bash().run({"command": "seq 1 100000; exit 7"}, ToolContext()).wait()
        self.assertIn("[truncated", out)
        self.assertTrue(out.endswith("Exit code: 7"))


if __name__ == "__main__":
    unittest.main()
