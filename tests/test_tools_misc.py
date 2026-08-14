"""Tests for the remaining tool modules: base (ToolContext/Registry),
Question, PlanExit, Skill, TodoWrite, Bash internals, and AgentTool.

These tools are normally exercised end-to-end through AgentSession
integration tests; this file drives them directly through their
injection points (session callbacks) so the interactive/async paths and
error containment boundaries are covered.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
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
    """Bash process-group kill fallbacks and communicate-failure path."""

    def test_kill_process_falls_back_to_proc_kill(self):
        from python_agent_harness.tools.bash import _kill_process

        proc = mock.Mock()
        proc.pid = 1234
        with (
            mock.patch("os.getpgid", return_value=42),
            mock.patch("os.killpg", side_effect=ProcessLookupError),
        ):
            _kill_process(proc)
        proc.kill.assert_called_once()

    def test_kill_process_swallows_kill_failure(self):
        from python_agent_harness.tools.bash import _kill_process

        proc = mock.Mock()
        proc.pid = 1234
        proc.kill.side_effect = ProcessLookupError
        with (
            mock.patch("os.getpgid", return_value=42),
            mock.patch("os.killpg", side_effect=PermissionError),
        ):
            _kill_process(proc)  # must not raise

    def test_communicate_failure_delivered_as_error(self):
        from python_agent_harness.tools.bash import Bash

        fake = mock.Mock()
        fake.communicate.side_effect = RuntimeError("pipe broke")
        with mock.patch("python_agent_harness.tools.bash.subprocess.Popen", return_value=fake):
            result = Bash().run({"command": "echo hi"}, ToolContext())
        self.assertIsInstance(result, PendingToolResult)
        self.assertIn("Error: Bash failed", result.wait())

    def test_popen_gets_devnull_stdin(self):
        """Bash must not inherit the harness's stdin: commands that read
        stdin would steal keystrokes from the TUI."""
        import subprocess

        from python_agent_harness.tools.bash import Bash

        fake = mock.Mock()
        fake.communicate.return_value = ("", None)
        with mock.patch(
            "python_agent_harness.tools.bash.subprocess.Popen", return_value=fake
        ) as popen:
            result = Bash().run({"command": "echo hi"}, ToolContext())
        result.wait()
        kwargs = popen.call_args.kwargs
        self.assertIs(kwargs["stdin"], subprocess.DEVNULL)


if __name__ == "__main__":
    unittest.main()
