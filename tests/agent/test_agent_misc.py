"""End-to-end agent loop tests with a fake client and fake session."""

import json
import os
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # tests/ for plan_cleanup

import plan_cleanup  # noqa: F401,E402  (side-effect: auto-remove /tmp plan dirs)
from agent_test_utils import (  # noqa: E402,F401
    FakeClient,
    ParallelToolSession,
    RealParallelSession,
    RecordingSession,
    SerialPromptSession,
    StaggeredSession,
    agent_call,
)

from python_agent_harness.agent import AgentLoop, sanitize_tool_result
from python_agent_harness.models import Message, ToolCall
from python_agent_harness.session import Session


class TestSanitizeToolResult(unittest.TestCase):
    def test_none_becomes_error_placeholder(self):
        self.assertEqual(
            sanitize_tool_result(None),
            "Error: tool produced no result (it may have been interrupted or failed to return).",
        )

    def test_empty_string_kept(self):
        self.assertEqual(sanitize_tool_result(""), "")

    def test_string_kept(self):
        self.assertEqual(sanitize_tool_result("x"), "x")

    def test_non_string_str_converted(self):
        self.assertEqual(sanitize_tool_result(42), "42")


class TestBashAsync(unittest.TestCase):
    """Async Bash contract: run() returns a PendingToolResult and the
    result is delivered when the process exits (mirrors :async t in
    gptel-agent-tools.el — the wait never blocks the sequential tool
    loop)."""

    def make_round(self, tmpdir):

        session = RecordingSession(project_dir=tmpdir)
        session.tools_enabled = False

        def real_execute(name, args, call_id=None):
            return Session.execute_tool(session, name, args, call_id=call_id)

        session.execute_tool = real_execute
        loop = AgentLoop(session, messages=[Message(role="user", content="run")])
        return session, loop

    def test_run_returns_pending_result(self):
        from python_agent_harness.tools import PendingToolResult, ToolContext
        from python_agent_harness.tools.bash import Bash

        with tempfile.TemporaryDirectory(prefix="pah-bash-") as tmpdir:
            session = RecordingSession(project_dir=tmpdir)
            result = Bash().run({"command": "echo hello"}, ToolContext(session))
            self.assertIsInstance(result, PendingToolResult)
            self.assertEqual(result.wait(), "hello\nExit code: 0")

    def test_round_delivers_async_result(self):
        with tempfile.TemporaryDirectory(prefix="pah-bash-") as tmpdir:
            with open(os.path.join(tmpdir, "x.txt"), "w") as f:
                f.write("file content\n")
            session, loop = self.make_round(tmpdir)
            loop.pending = [
                ToolCall(id="b1", name="Bash", arguments=json.dumps({"command": "echo one"})),
                ToolCall(
                    id="b2",
                    name="Read",
                    arguments=json.dumps({"file_path": os.path.join(tmpdir, "x.txt")}),
                ),
            ]
            loop._run_tool_round()
            by_id = {m.tool_call_id: m.text().strip() for m in loop.messages if m.role == "tool"}
            self.assertEqual(by_id["b1"], "one\nExit code: 0")
            # sync sibling delivered alongside the async one
            self.assertEqual(by_id["b2"], "file content")
            self.assertEqual(
                [m.tool_call_id for m in loop.messages if m.role == "tool"],
                ["b1", "b2"],  # original order preserved
            )

    def test_cancel_kills_process_and_delivers_error(self):
        from python_agent_harness.tools import PendingToolResult, ToolContext
        from python_agent_harness.tools.bash import Bash

        with tempfile.TemporaryDirectory(prefix="pah-bash-") as tmpdir:
            session = RecordingSession(project_dir=tmpdir)
            result = Bash().run({"command": "sleep 30"}, ToolContext(session))
            self.assertIsInstance(result, PendingToolResult)
            threading.Timer(0.5, session.cancel).start()
            start = time.monotonic()
            delivered = result.wait()
            elapsed = time.monotonic() - start
            self.assertLess(elapsed, 5)  # killpg unblocked the wait promptly
            self.assertIn("cancelled", delivered)

    def test_deliver_is_idempotent(self):
        from python_agent_harness.tools import PendingToolResult

        p = PendingToolResult()
        p.deliver("first")
        p.deliver("second")  # late duplicate must be a no-op
        self.assertEqual(p.wait(), "first")

    def test_bad_command_returns_error_string_not_pending(self):
        from python_agent_harness.tools import ToolContext
        from python_agent_harness.tools.bash import Bash

        # Popen with shell=True never fails on syntax; simulate the
        # OSError path via an impossible cwd instead
        class FakeSess:
            project_dir = "/nonexistent-pah-dir"

        result = Bash().run({"command": "echo hi"}, ToolContext(FakeSess()))
        self.assertIsInstance(result, str)
        self.assertTrue(result.startswith("Error:"))


class TestAutoSave(unittest.TestCase):
    """Auto-save failures must not be silent: retry once, then leave a
    persistent, visible error state (cleared by the next success)."""

    def make_session(self):
        session = RecordingSession()
        session.logs = []
        session.log_fn = session.logs.append
        session.notified = []
        session.notify_fn = lambda kind, data=None: session.notified.append(kind)
        return session

    def test_retry_then_persistent_error(self):
        session = self.make_session()
        fails = {"n": 0}

        def flaky_save(text):
            fails["n"] += 1
            if fails["n"] <= 2:
                raise OSError("disk full")
            return None

        with mock.patch.object(session.store, "save", side_effect=flaky_save):
            session.auto_save([Message(role="user", content="hi")], None)
        self.assertEqual(fails["n"], 2)  # retried once, failed again
        self.assertEqual(session._save_error, "disk full")
        self.assertIn("auto-save failed", session.logs[-1])
        self.assertIn("save-error", session.notified)
        # the next successful save clears the persistent error
        with mock.patch.object(session.store, "save", return_value=None):
            session.auto_save([Message(role="user", content="hi")], None)
        self.assertIsNone(session._save_error)

    def test_transient_failure_recovers(self):
        """A one-off failure (retry succeeds) must not leave an error
        state behind."""
        session = self.make_session()
        fails = {"n": 0}

        def transient_save(text):
            fails["n"] += 1
            if fails["n"] == 1:
                raise OSError("nfs hiccup")
            return None

        with mock.patch.object(session.store, "save", side_effect=transient_save):
            session.auto_save([Message(role="user", content="hi")], None)
        self.assertEqual(fails["n"], 2)  # retried
        self.assertIsNone(session._save_error)  # success cleared it
        self.assertNotIn("save-error", session.notified)
