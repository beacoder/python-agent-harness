"""End-to-end agent loop tests with a fake client and fake session."""

import contextlib
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

from python_agent_harness import config
from python_agent_harness.agent import AgentLoop
from python_agent_harness.models import Message, ToolCall, Usage
from python_agent_harness.planmode import PlanMode
from python_agent_harness.session import Session
from python_agent_harness.tools import default_registry


class TestAgentLoop(unittest.TestCase):
    def test_simple_turn(self):
        session = RecordingSession()
        session.tools_enabled = False  # non-agentic: no completion nudges
        session.client.script = ["hello there"]
        loop = AgentLoop(session, messages=[Message(role="user", content="hi")])
        result = loop.run()
        self.assertEqual(result, "hello there")

    def test_subagent_budget_exhausted_returns_last_real_text(self):
        """Round-budget exhaustion must surface the last real assistant
        text, never a trailing tool result or an empty string."""
        session = RecordingSession()
        session.tools_enabled = False
        # round 1: text + tool call; round 2: tool call only; budget 2
        session.client.script = [
            (
                "partial answer",
                [ToolCall(id="1", name="Read", arguments='{"file_path": "/tmp/x.py"}')],
            ),
            ("", [ToolCall(id="2", name="Read", arguments='{"file_path": "/tmp/x.py"}')]),
        ]
        loop = AgentLoop(
            session,
            messages=[Message(role="user", content="do it")],
            top_level=False,
            max_rounds=2,
        )
        result = loop.run()
        # the trailing messages are [tool result, assistant+tool_call,
        # tool result]; the only real text is "partial answer"
        self.assertEqual(result, "partial answer")

    def test_subagent_budget_exhausted_no_text_returns_error(self):
        """Exhaustion with NO assistant text at all (only tool rounds)
        must return an informative error, not raw tool output or ''."""
        session = RecordingSession()
        session.tools_enabled = False
        session.client.script = [
            ("", [ToolCall(id="1", name="Read", arguments='{"file_path": "/tmp/x.py"}')]),
            ("", [ToolCall(id="2", name="Read", arguments='{"file_path": "/tmp/x.py"}')]),
        ]
        loop = AgentLoop(
            session,
            messages=[Message(role="user", content="do it")],
            top_level=False,
            max_rounds=2,
        )
        result = loop.run()
        self.assertIsInstance(result, str)
        self.assertTrue(result.startswith("Error: sub-agent round budget"), result)
        self.assertNotEqual(result, "file content")  # not raw tool output
        self.assertNotEqual(result, "")

    def test_tool_round(self):
        session = RecordingSession()
        with mock.patch("python_agent_harness.config.MAX_NUDGES", 1):
            session.client.script = [
                ("", [ToolCall(id="1", name="Read", arguments='{"file_path": "/tmp/x.py"}')]),
                "final answer",
                "final answer",
            ]
            loop = AgentLoop(session, messages=[Message(role="user", content="read it")])
            result = loop.run()
        self.assertEqual(result, "final answer")
        self.assertEqual(session.executed[0][0], "Read")
        self.assertTrue(any(m.role == "tool" and m.text() == "file content" for m in loop.messages))

    def test_sync_tools_run_one_by_one_in_call_order(self):
        """Sync tools in one round — everything except the async
        Bash/Agent — execute ONE AT A TIME in model-emitted order
        (gptel-style): peak concurrency stays 1 and the round takes
        ~the sum of the durations, not one duration.

        Uses a mix of readonly (Read, Grep) and non-readonly
        (TodoWrite) tools so the round takes the sequential path
        (a non-readonly tool in the round forces sequential dispatch)."""
        session = ParallelToolSession(duration=0.2)
        session.tools_enabled = False
        session.client.script = [
            (
                "",
                [
                    ToolCall(id="1", name="Read", arguments='{"file_path": "/tmp/x.py"}'),
                    ToolCall(id="2", name="TodoWrite", arguments='{"todos": []}'),
                    ToolCall(id="3", name="Grep", arguments='{"regex": "x", "path": "/tmp"}'),
                ],
            ),
            "all done",
        ]
        loop = AgentLoop(session, messages=[Message(role="user", content="delegate")])
        start = time.monotonic()
        result = loop.run()
        elapsed = time.monotonic() - start
        self.assertEqual(result, "all done")
        # sequential: never more than one tool at a time...
        self.assertEqual(session.max_active, 1)
        # ...and the round took ~3 x 0.2s, not one 0.2s batch
        self.assertGreaterEqual(elapsed, 0.55)
        # results delivered in original call order
        tool_rows = [m.text() for m in loop.messages if m.role == "tool"]
        self.assertEqual(
            tool_rows,
            ["result of Read", "result of TodoWrite", "result of Grep"],
        )

    def test_async_bash_overlaps_with_sequential_sync_tools(self):
        """Bash is async (gptel `:async t`): a real Bash call is
        dispatched and runs in the background while the following sync
        tool executes inline, so the round finishes in ~the Bash
        duration, not the sum."""
        with tempfile.TemporaryDirectory(prefix="pah-bash-") as tmpdir:
            with open(os.path.join(tmpdir, "x.txt"), "w") as f:
                f.write("file content\n")
            session = RecordingSession(project_dir=tmpdir)
            session.tools_enabled = False

            def real_execute(name, args, call_id=None):
                return Session.execute_tool(session, name, args, call_id=call_id)

            session.execute_tool = real_execute
            loop = AgentLoop(session, messages=[Message(role="user", content="run")])
            calls = [
                ToolCall(
                    id="b1", name="Bash", arguments=json.dumps({"command": "sleep 0.5 && echo one"})
                ),
                ToolCall(
                    id="b2",
                    name="Read",
                    arguments=json.dumps({"file_path": os.path.join(tmpdir, "x.txt")}),
                ),
            ]
            loop.pending = list(calls)
            start = time.monotonic()
            loop._run_tool_round()
            elapsed = time.monotonic() - start
            by_id = {m.tool_call_id: m.text().strip() for m in loop.messages if m.role == "tool"}
            self.assertEqual(by_id["b1"], "one\nExit code: 0")
            self.assertEqual(by_id["b2"], "file content")
            # ~0.5s if the Bash ran in the background during the Read,
            # ~0.6s+ if the Read waited for the Bash to finish
            self.assertLess(elapsed, 0.9)
            # delivered in original order
            self.assertEqual(
                [m.tool_call_id for m in loop.messages if m.role == "tool"],
                ["b1", "b2"],
            )

    def test_mixed_round_sync_tools_serialize_async_tools_overlap(self):
        """A mixed round: Agent and Bash (async) are dispatched and
        overlap each other, while sync tools execute one by one; all
        results are delivered in original order."""
        session = ParallelToolSession(duration=0.3)
        session.tools_enabled = False
        session.client.script = [
            (
                "",
                [
                    agent_call("1", "alpha"),
                    ToolCall(id="2", name="Read", arguments='{"file_path": "/tmp/x.py"}'),
                    agent_call("3", "beta"),
                ],
            ),
            "done",
        ]
        loop = AgentLoop(session, messages=[Message(role="user", content="go")])
        result = loop.run()
        self.assertEqual(result, "done")
        # the fake session treats every tool as sync (no
        # PendingToolResult), so the round is fully sequential
        self.assertEqual(session.max_active, 1)
        by_id = {m.tool_call_id: m.text() for m in loop.messages if m.role == "tool"}
        self.assertEqual(by_id["1"], "done:alpha")
        self.assertEqual(by_id["2"], "result of Read")
        self.assertEqual(by_id["3"], "done:beta")
        # delivered in original order
        self.assertEqual(
            [m.tool_call_id for m in loop.messages if m.role == "tool"],
            ["1", "2", "3"],
        )

    def test_cancel_during_tool_round(self):
        """Ctrl-C while a long tool call runs: the round stops, the
        remaining calls are skipped, the run returns None, and no
        exception escapes."""
        session = ParallelToolSession(duration=None)
        session.tools_enabled = False
        session.client.script = [
            ("", [agent_call("1", "alpha"), agent_call("2", "beta")]),
        ]
        result = {}
        worker = threading.Thread(
            target=lambda: result.update(
                r=AgentLoop(session, messages=[Message(role="user", content="delegate")]).run()
            )
        )
        worker.start()
        self.assertTrue(session.started.wait(timeout=5))
        time.sleep(0.2)  # call 1 is now running
        session.cancel()
        worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.assertIsNone(result.get("r"))

    def test_cancelled_queued_tool_does_not_run(self):
        """A tool still QUEUED when Ctrl-C lands must never execute:
        call 1 blocks, the cancel arrives while it runs, and the
        not-yet-started call 2 is skipped by the sequential loop's
        per-call pre-check (deterministic counterpart of the salvage
        tests' 2-3 range).

        Uses a non-readonly tool (TodoWrite) so the round takes the
        sequential path (readonly-only rounds dispatch in parallel)."""
        session = ParallelToolSession(duration=None)
        session.tools_enabled = False
        session.client.script = [
            (
                "",
                [
                    ToolCall(id="1", name="Read", arguments='{"file_path": "/tmp/a.py"}'),
                    ToolCall(id="2", name="TodoWrite", arguments='{"todos": []}'),
                ],
            ),
        ]
        result = {}
        worker = threading.Thread(
            target=lambda: result.update(
                r=AgentLoop(session, messages=[Message(role="user", content="read")]).run()
            )
        )
        worker.start()
        self.assertTrue(session.started.wait(timeout=5))
        time.sleep(0.2)  # call 1 is now blocked; call 2 has not started
        session.cancel()
        worker.join(timeout=10)
        self.assertFalse(worker.is_alive())
        self.assertIsNone(result.get("r"))
        # only the already-running call executed; the not-yet-started
        # call was skipped after Ctrl-C and never reached execute_tool
        self.assertEqual(session.executed_count, 1)
        self.assertEqual(session.max_active, 1)

    def test_all_sync_calls_run_sequentially(self):
        """Every sync tool call in a round executes (nothing is
        dropped), one at a time in call order: peak concurrency stays
        1 and the round takes ~the sum of the durations.

        Uses a non-readonly tool (TodoWrite) so the round takes the
        sequential path (readonly-only rounds dispatch in parallel)."""
        session = ParallelToolSession(duration=0.15)
        session.tools_enabled = False
        calls = [
            ToolCall(id=str(i), name="TodoWrite", arguments='{"todos": []}') for i in range(1, 9)
        ]
        session.client.script = [("", calls), "done"]
        loop = AgentLoop(session, messages=[Message(role="user", content="go")])
        start = time.monotonic()
        result = loop.run()
        elapsed = time.monotonic() - start
        self.assertEqual(result, "done")
        # all 8 calls ran, one at a time
        self.assertEqual(session.executed_count, 8)
        self.assertEqual(session.max_active, 1)
        # ...so the round took ~8 x 0.15s, not one 0.15s batch
        self.assertGreaterEqual(elapsed, 1.0)
        # results delivered in original order
        self.assertEqual(
            [m.tool_call_id for m in loop.messages if m.role == "tool"],
            [str(i) for i in range(1, 9)],
        )

    def test_malformed_tool_arguments_do_not_break_round(self):
        """A tool call whose arguments are not valid JSON (or parse to a
        non-object) must not break the round: the call gets an error
        result, sibling calls still run, and all results are delivered."""
        session = RecordingSession()
        session.tools_enabled = False
        session.client.script = [
            (
                "",
                [
                    ToolCall(id="1", name="Read", arguments='{"file_path": "/tmp/a.py"}'),
                    ToolCall(id="2", name="Bash", arguments="not-json{{{"),
                    ToolCall(id="3", name="Read", arguments='{"file_path": "/tmp/c.py"}'),
                    ToolCall(id="4", name="Read", arguments="[1, 2, 3]"),
                ],
            ),
            "done",
        ]
        loop = AgentLoop(session, messages=[Message(role="user", content="go")])
        result = loop.run()
        self.assertEqual(result, "done")
        # only the well-formed calls reached execute_tool
        read_args = [args for name, args in session.executed if name == "Read"]
        bash_args = [args for name, args in session.executed if name == "Bash"]
        self.assertEqual(
            sorted(str(a["file_path"]) for a in read_args),
            ["/tmp/a.py", "/tmp/c.py"],
        )
        # malformed calls never reached the tool
        self.assertEqual(bash_args, [])
        # all four results delivered in original order
        self.assertEqual(
            [m.tool_call_id for m in loop.messages if m.role == "tool"],
            ["1", "2", "3", "4"],
        )
        by_id = {m.tool_call_id: m.text() for m in loop.messages if m.role == "tool"}
        self.assertEqual(by_id["1"], "file content")
        self.assertEqual(by_id["3"], "file content")
        # malformed args get clear error messages
        self.assertIn("missing required argument", by_id["2"])
        self.assertIn("missing required argument", by_id["4"])

    def test_tool_round_contains_tool_crash(self):
        """A tool that raises inside a round must not kill the round:
        sibling tools still run, the crash becomes an error result,
        and all results are delivered in original order."""
        session = RecordingSession()
        session.tools_enabled = False
        orig_execute = RecordingSession.execute_tool

        def exploding_execute(name, args, call_id=None):
            if name == "Grep":
                raise RuntimeError("boom")
            return orig_execute(session, name, args, call_id=call_id)

        session.execute_tool = exploding_execute
        session.client.script = [
            (
                "",
                [
                    ToolCall(id="1", name="Read", arguments='{"file_path": "/tmp/a.py"}'),
                    ToolCall(id="2", name="Bash", arguments='{"command": "echo hi"}'),
                    ToolCall(id="3", name="Grep", arguments='{"regex": "x", "path": "/tmp"}'),
                ],
            ),
            "all done",
        ]
        loop = AgentLoop(session, messages=[Message(role="user", content="go")])
        self.assertEqual(loop.run(), "all done")
        # the two healthy tools ran; the crash was contained
        self.assertEqual(sorted(n for n, _ in session.executed), ["Bash", "Read"])
        tool_rows = [(m.tool_call_id, m.text()) for m in loop.messages if m.role == "tool"]
        self.assertEqual(tool_rows[0], ("1", "file content"))
        self.assertEqual(tool_rows[1], ("2", "bash output"))
        self.assertEqual(tool_rows[2][0], "3")
        self.assertIn("crashed during execution", tool_rows[2][1])
        self.assertIn("boom", tool_rows[2][1])

    def test_interactive_prompts_serialized(self):
        """Interactive prompts (Question tool, PlanExit confirmation)
        must be strictly serialized: concurrent calls never present more
        than one prompt at a time."""
        session = SerialPromptSession(duration=0.3)
        threads = [
            threading.Thread(target=lambda: session.ask_questions([{"question": "q1"}])),
            threading.Thread(target=lambda: session.confirm("switch?")),
            threading.Thread(target=lambda: session.ask_questions([{"question": "q2"}])),
        ]
        start = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.monotonic() - start
        # 3 prompts x 0.3s, one at a time
        self.assertEqual(session.max_active, 1)
        self.assertGreaterEqual(elapsed, 0.6)

    def test_edits_attach_diffs_to_their_own_call(self):
        """Two real Edit calls in one round must attach each unified
        diff to ITS OWN tool call: the thread-local diff slot (not a
        shared one) is what keeps file mutations from cross-attributing
        diffs in the TUI."""
        import tempfile as _tf

        with _tf.TemporaryDirectory(prefix="pah-parallel-diff-") as tmpdir:
            fa = os.path.join(tmpdir, "a.py")
            fb = os.path.join(tmpdir, "b.py")
            with open(fa, "w") as f:
                f.write("x = 1\n")
            with open(fb, "w") as f:
                f.write("y = 2\n")
            session = RecordingSession(project_dir=tmpdir)
            session.tools_enabled = False

            # run the REAL Edit implementation (RecordingSession normally
            # returns canned results)
            def real_execute(name, args, call_id=None):
                return Session.execute_tool(session, name, args, call_id=call_id)

            session.execute_tool = real_execute
            loop = AgentLoop(session, messages=[Message(role="user", content="edit")])
            calls = [
                ToolCall(
                    id="e1",
                    name="Edit",
                    arguments=json.dumps({"path": fa, "old_str": "x = 1", "new_str": "x = 42"}),
                ),
                ToolCall(
                    id="e2",
                    name="Edit",
                    arguments=json.dumps({"path": fb, "old_str": "y = 2", "new_str": "y = 43"}),
                ),
            ]
            loop.pending = list(calls)
            loop._run_tool_round()
            # both files really changed
            with open(fa) as f:
                self.assertEqual(f.read(), "x = 42\n")
            with open(fb) as f:
                self.assertEqual(f.read(), "y = 43\n")
            # each call carries its own file's diff — not the other's
            self.assertIn("a.py", calls[0].diff)
            self.assertIn("x = 42", calls[0].diff)
            self.assertNotIn("b.py", calls[0].diff)
            self.assertIn("b.py", calls[1].diff)
            self.assertIn("y = 43", calls[1].diff)
            self.assertNotIn("a.py", calls[1].diff)
            # results delivered in original order
            self.assertEqual(
                [m.tool_call_id for m in loop.messages if m.role == "tool"],
                ["e1", "e2"],
            )

    def test_none_result_becomes_placeholder_in_tool_round(self):
        """A tool returning None inside a tool round must yield the
        NIL placeholder (never a crash or a missing tool row), while
        sibling results are delivered normally."""
        session = RecordingSession()
        session.tools_enabled = False
        orig_execute = RecordingSession.execute_tool

        def none_execute(name, args, call_id=None):
            if name == "Bash":
                return None
            return orig_execute(session, name, args, call_id=call_id)

        session.execute_tool = none_execute
        session.client.script = [
            (
                "",
                [
                    ToolCall(id="1", name="Read", arguments='{"file_path": "/tmp/a.py"}'),
                    ToolCall(id="2", name="Bash", arguments='{"command": "echo hi"}'),
                    ToolCall(id="3", name="Read", arguments='{"file_path": "/tmp/c.py"}'),
                ],
            ),
            "done",
        ]
        loop = AgentLoop(session, messages=[Message(role="user", content="go")])
        self.assertEqual(loop.run(), "done")
        by_id = {m.tool_call_id: m.text() for m in loop.messages if m.role == "tool"}
        self.assertEqual(by_id["1"], "file content")
        self.assertIn("produced no result", by_id["2"])
        self.assertEqual(by_id["3"], "file content")
        self.assertEqual(
            [m.tool_call_id for m in loop.messages if m.role == "tool"],
            ["1", "2", "3"],
        )

    def test_plan_mode_blocks_writes_in_tool_round(self):
        """A tool round under plan mode: mutating tools are blocked
        by the read-only guard while read-only tools run — every result
        is still delivered in original order, and no forbidden write
        lands."""
        session = RecordingSession()
        session.tools_enabled = False
        session.plan_mode = PlanMode("/tmp/fakeproj")
        session.plan_mode.set_mode(
            session.plan_mode.mode.PLAN,
            {
                "plan": "P1",
                "plan-mode": "P2",
                "build-switch": "B",
            },
        )

        def real_execute(name, args, call_id=None):
            return Session.execute_tool(session, name, args, call_id=call_id)

        session.execute_tool = real_execute
        with tempfile.TemporaryDirectory(prefix="pah-plan-") as tmpdir:
            blocked_path = os.path.join(tmpdir, "blocked.txt")
            readable = os.path.join(tmpdir, "readable.txt")
            with open(readable, "w") as f:
                f.write("readable content\n")
            session.client.script = [
                (
                    "",
                    [
                        ToolCall(
                            id="1",
                            name="Write",
                            arguments=json.dumps(
                                {
                                    "path": tmpdir,
                                    "filename": "blocked.txt",
                                    "content": "should not land",
                                }
                            ),
                        ),
                        ToolCall(
                            id="2", name="Read", arguments=json.dumps({"file_path": readable})
                        ),
                    ],
                ),
                "done",
            ]
            loop = AgentLoop(session, messages=[Message(role="user", content="go")])
            self.assertEqual(loop.run(), "done")
            # the Write was blocked by plan mode; the Read still ran
            self.assertFalse(os.path.exists(blocked_path))
            by_id = {m.tool_call_id: m.text() for m in loop.messages if m.role == "tool"}
            self.assertIn("blocked by plan mode", by_id["1"])
            self.assertEqual(by_id["2"], "readable content\n")
        self.assertEqual(
            [m.tool_call_id for m in loop.messages if m.role == "tool"],
            ["1", "2"],
        )

    def test_real_bash_commands_run_concurrently(self):
        """Two REAL Bash invocations in one round complete in ~one sleep
        duration, not two: async tools are dispatched and overlap
        (gptel `:async t` behavior)."""
        with tempfile.TemporaryDirectory(prefix="pah-bash-") as tmpdir:
            session = RecordingSession(project_dir=tmpdir)
            session.tools_enabled = False

            def real_execute(name, args, call_id=None):
                return Session.execute_tool(session, name, args, call_id=call_id)

            session.execute_tool = real_execute
            loop = AgentLoop(session, messages=[Message(role="user", content="run")])
            calls = [
                ToolCall(
                    id="b1", name="Bash", arguments=json.dumps({"command": "sleep 0.5 && echo one"})
                ),
                ToolCall(
                    id="b2", name="Bash", arguments=json.dumps({"command": "sleep 0.5 && echo two"})
                ),
            ]
            loop.pending = list(calls)
            start = time.monotonic()
            loop._run_tool_round()
            elapsed = time.monotonic() - start
            by_id = {m.tool_call_id: m.text().strip() for m in loop.messages if m.role == "tool"}
            self.assertEqual(by_id["b1"], "one\nExit code: 0")
            self.assertEqual(by_id["b2"], "two\nExit code: 0")
            # ~0.5s if the subprocesses ran concurrently, ~1.0s if serialized
            self.assertLess(elapsed, 0.9)

    def test_nudge_redirect(self):
        session = RecordingSession()
        with mock.patch("python_agent_harness.config.MAX_NUDGES", 1):
            session.client.script = ["almost done", "done now"]
            loop = AgentLoop(session, messages=[Message(role="user", content="do it")])
            result = loop.run()
        self.assertEqual(result, "done now")
        # the nudge message was injected
        self.assertTrue(
            any(m.role == "user" and "Task Completion Rules" in m.text() for m in loop.messages)
        )

    def test_plan_mode_queues_prompts(self):
        session = RecordingSession()
        session.client.script = ["ok"]
        session.plan_mode = PlanMode("/tmp/fakeproj")
        session.plan_mode.set_mode(
            session.plan_mode.mode.PLAN,
            {
                "plan": "P1",
                "plan-mode": "P2 ${planInfo}",
                "build-switch": "B",
            },
        )
        loop = AgentLoop(session, messages=[Message(role="user", content="plan it")])
        loop.run()
        sent = session.client.calls[0]
        contents = [m.get("content") for m in sent if m.get("role") == "user"]
        self.assertIn("P1", contents)
        self.assertTrue(any("P2 " in c for c in contents))

    def test_compact_on_high_context(self):
        session = RecordingSession()
        session.client.script = ["answer after compaction"]
        with mock.patch("python_agent_harness.config.MAX_NUDGES", 0):
            calls = {"n": 0}

            def fake_estimate(*a, **k):
                calls["n"] += 1
                return 1_000_000 if calls["n"] <= 2 else 100

            with mock.patch(
                "python_agent_harness.agent.estimate_payload_tokens",
                side_effect=fake_estimate,
            ):
                loop = AgentLoop(session, messages=[Message(role="user", content="long task")])
                result = loop.run()
        self.assertEqual(result, "answer after compaction")
        # compaction replaced the conversation with summary frame + request
        # as USER messages (the system prompt stays untouched)
        self.assertEqual(loop.messages[0].role, "user")
        self.assertEqual(loop.messages[1].role, "user")
        self.assertIn("Compacted Summary", loop.messages[0].text())
        self.assertTrue(any("Compacted Summary" in m.text() for m in loop.messages))

    def test_compact_resets_nudge_budget(self):
        """Compaction must reset the nudge budget: a terminal answer right
        after compaction must not end the run just because the budget was
        spent before compaction (elisp parity)."""
        session = RecordingSession()
        session.client.script = [
            "premature final answer 1",
            "premature final answer 2",
            "premature final answer 3",
            "answer after compaction",
        ]
        session.tools_enabled = True
        calls = {"n": 0}

        def fake_estimate(*a, **k):
            calls["n"] += 1
            return 1_000_000 if calls["n"] == 3 else 100

        with mock.patch(
            "python_agent_harness.agent.estimate_payload_tokens",
            side_effect=fake_estimate,
        ):
            loop = AgentLoop(session, messages=[Message(role="user", content="long task")])
            result = loop.run()
        self.assertEqual(result, "done")
        self.assertEqual(loop.messages[0].role, "user")
        self.assertIn("Compacted Summary", loop.messages[0].text())
        self.assertGreater(loop.supervisor.nudge_count, 0)

    def test_manual_compact_keeps_every_user_prompt(self):
        """Manual /compact replaces the history with the summary frame
        followed by every real user prompt: the compacted info and the
        actual requests both survive, so the model can still see the
        tasks (nudges and other harness-injected messages excluded)."""
        session = RecordingSession()
        session.tools_enabled = False
        session.last_messages = [
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi"),
            Message(role="user", content="please continue"),
            Message(role="user", content=config.NUDGE_MESSAGE, injected=True),
        ]
        ok, msg = session.compact_conversation()
        self.assertTrue(ok)
        self.assertEqual([m.role for m in session.last_messages], ["user", "user", "user"])
        self.assertTrue(session.last_messages[0].text().startswith("**[Compacted Summary]**"))
        self.assertIn("SYNC-OK", session.last_messages[0].text())
        self.assertEqual(
            [m.text() for m in session.last_messages[1:]], ["hello", "please continue"]
        )

    def test_manual_compact_does_not_require_user_request(self):
        """Compacting a summary-only history (e.g. a second /compact)
        must still work: the manual command no longer needs a user
        request to resume with."""
        session = RecordingSession()
        session.tools_enabled = False
        session.last_messages = [
            Message(role="user", content="**[Compacted Summary]**\n\nold summary"),
        ]
        ok, _ = session.compact_conversation()
        self.assertTrue(ok)
        self.assertEqual(len(session.last_messages), 1)
        self.assertEqual(session.last_messages[0].role, "user")
        self.assertIn("Compacted Summary", session.last_messages[0].text())

    def test_auto_compact_mirrors_shared_history(self):
        """The in-loop compaction must mirror the compacted conversation
        onto session.last_messages, so the TUI and a later manual
        /compact see the summary — not the old full history."""
        session = RecordingSession()
        session.client.script = ["answer after compaction"]
        with mock.patch("python_agent_harness.config.MAX_NUDGES", 0):
            calls = {"n": 0}

            def fake_estimate(*a, **k):
                calls["n"] += 1
                return 1_000_000 if calls["n"] <= 2 else 100

            with mock.patch(
                "python_agent_harness.agent.estimate_payload_tokens",
                side_effect=fake_estimate,
            ):
                loop = AgentLoop(session, messages=[Message(role="user", content="long task")])
                loop.run()
        # the compacted frame reached the shared history as a user message
        self.assertEqual(session.last_messages[0].role, "user")
        self.assertIn("Compacted Summary", session.last_messages[0].text())

    def test_auto_compact_keeps_every_user_prompt(self):
        """In-loop compaction preserves EVERY real user prompt after
        the summary frame (nudges and other harness-injected messages
        excluded), not just the last one — and repeated compaction
        never stacks old summary frames or duplicates prompts."""
        session = RecordingSession()
        session.client.script = ["answer after compaction"]
        with mock.patch("python_agent_harness.config.MAX_NUDGES", 0):
            calls = {"n": 0}

            def fake_estimate(*a, **k):
                calls["n"] += 1
                return 1_000_000 if calls["n"] <= 2 else 100

            with mock.patch(
                "python_agent_harness.agent.estimate_payload_tokens",
                side_effect=fake_estimate,
            ):
                loop = AgentLoop(
                    session,
                    messages=[
                        Message(role="user", content="task one"),
                        Message(role="assistant", content="done one"),
                        Message(role="user", content="task two"),
                    ],
                )
                result = loop.run()
        self.assertEqual(result, "answer after compaction")
        # the history is [summary frame, task one, task two, final
        # answer]: the old assistant turn is gone, every real prompt
        # survives, and the second compaction (the fake ratio stays
        # high for two rounds) did not stack another frame
        self.assertEqual(len(loop.messages), 4)
        self.assertTrue(loop.messages[0].text().startswith("**[Compacted Summary]**"))
        self.assertEqual([m.text() for m in loop.messages[1:3]], ["task one", "task two"])
        self.assertEqual(loop.messages[3].text(), "answer after compaction")
        # the shared history mirrors the compacted conversation
        self.assertEqual(
            [m.text() for m in session.last_messages], [m.text() for m in loop.messages]
        )

    def test_auto_save_and_last_messages(self):
        session = RecordingSession()
        session.tools_enabled = False
        session.client.script = ["bye"]
        loop = AgentLoop(session, messages=[Message(role="user", content="hi")])
        loop.run()
        self.assertTrue(session.last_messages)
        self.assertTrue(session.store.file_path)

    def test_reasoning_effort_reaches_client(self):
        session = RecordingSession()
        session.tools_enabled = False
        session.reasoning_effort = "high"
        session.client.script = ["ok"]
        loop = AgentLoop(session, messages=[Message(role="user", content="hi")])
        loop.run()
        # FakeClient.chat records the tool kwargs; verify it was passed
        last_call = session.client.kwargs[-1]
        self.assertEqual(last_call.get("reasoning_effort"), "high")

    def test_reasoning_effort_none_omitted(self):
        session = RecordingSession()
        session.tools_enabled = False
        session.reasoning_effort = None
        session.client.script = ["ok"]
        loop = AgentLoop(session, messages=[Message(role="user", content="hi")])
        loop.run()
        last_call = session.client.kwargs[-1]
        self.assertIsNone(last_call.get("reasoning_effort"))

    def test_stream_defaults_true_on_session_and_client(self):
        """Streaming is the default: the session opts in unless the
        config file (or --no-stream) says otherwise, and the loop must
        forward that to the client."""
        session = RecordingSession()
        session.tools_enabled = False
        session.client.script = ["ok"]
        loop = AgentLoop(session, messages=[Message(role="user", content="hi")])
        loop.run()
        self.assertIs(session.stream, True)
        self.assertIs(session.client.kwargs[-1]["stream"], True)

    def test_non_streaming_reaches_client(self):
        """A session configured non-streaming must send stream=False to
        the client and still complete the loop normally."""
        session = RecordingSession()
        session.stream = False
        session.tools_enabled = False
        session.client.script = ["ok"]
        loop = AgentLoop(session, messages=[Message(role="user", content="hi")])
        result = loop.run()
        self.assertEqual(result, "ok")
        self.assertIs(session.client.kwargs[-1]["stream"], False)

    def test_non_streaming_tool_round(self):
        """Non-streaming mode must support full tool rounds (the client
        parses tool_calls from the single response)."""
        session = RecordingSession()
        session.stream = False
        with mock.patch("python_agent_harness.config.MAX_NUDGES", 1):
            session.client.script = [
                ("", [ToolCall(id="1", name="Read", arguments='{"file_path": "/tmp/x.py"}')]),
                "final answer",
                "final answer",
            ]
            loop = AgentLoop(session, messages=[Message(role="user", content="read it")])
            result = loop.run()
        self.assertEqual(result, "final answer")
        self.assertEqual(session.executed[0][0], "Read")
        self.assertTrue(any(m.role == "tool" and m.text() == "file content" for m in loop.messages))
        # every request in the run went out non-streaming
        self.assertTrue(all(k["stream"] is False for k in session.client.kwargs))

    def test_non_streaming_http_tool_round_end_to_end(self):
        """Non-streaming through the REAL HTTP client: tool_calls parsed
        from a single response drive a full tool round, the loop finishes
        with the final answer, and every request went out stream=False."""
        import tempfile
        from pathlib import Path

        import fake_openai_server
        from fake_openai_server import serve

        import python_agent_harness.config as cfg
        from python_agent_harness.client import Client

        with tempfile.TemporaryDirectory() as d:
            cfg.SESSION_DIR = Path(d)  # session store writes land in tmp
            data_file = Path(d) / "data.txt"
            data_file.write_text("hello data", encoding="utf-8")
            fake_openai_server.reset_state()
            try:
                fake_openai_server.NON_STREAM_SEQUENCE = [
                    {  # 1st request: a tool call, no text
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "",
                                    "tool_calls": [
                                        {
                                            "id": "call_1",
                                            "type": "function",
                                            "function": {
                                                "name": "Read",
                                                "arguments": json.dumps(
                                                    {"file_path": str(data_file)}
                                                ),
                                            },
                                        }
                                    ],
                                }
                            }
                        ],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 2},
                    },
                    {  # 2nd request: the final answer
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "http non-streaming done",
                                }
                            }
                        ],
                        "usage": {"prompt_tokens": 3, "completion_tokens": 4},
                    },
                ]
                srv = serve()
                host, port = srv.server_address
                client = Client(base_url=f"http://{host}:{port}/v1", api_key="test", model="fake")
                session = Session(
                    project_dir=d,
                    client=client,
                    model="fake",
                    registry=default_registry(),
                    stream=False,
                )
                # non-agentic: no completion nudges — the loop must terminate
                # on the scripted final answer; the fake server still returns
                # tool_calls, so the tool round runs regardless
                session.tools_enabled = False
                try:
                    loop = AgentLoop(session, messages=[Message(role="user", content="read it")])
                    result = loop.run()
                finally:
                    client.close()
                # snapshot the bodies before resetting shared server state
                bodies = list(fake_openai_server.REQUEST_BODIES)
            finally:
                fake_openai_server.reset_state()  # don't leak server state
            self.assertEqual(result, "http non-streaming done")
            # the tool round really executed against the HTTP response
            tool_rows = [m for m in loop.messages if m.role == "tool"]
            self.assertEqual(len(tool_rows), 1)
            self.assertIn("hello data", tool_rows[0].text())
            # every request (loop chats + title generation) went out
            # non-streaming with the stream flag set
            self.assertTrue(bodies)
            self.assertTrue(all(b.get("stream") is False for b in bodies))

    def test_abort_unblocks_stalled_http_stream(self):
        """Ctrl-C during a REAL stalled chunked SSE stream must unblock
        the worker promptly: closing the pool is not enough on Linux
        (close() cannot wake a blocked recv), so abort() must shutdown
        the in-flight connection socket.  Regression test for Ctrl-C
        leaving a zombie worker that would stall history adoption."""
        import tempfile
        import threading as _threading
        import time as _time
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from pathlib import Path

        import python_agent_harness.config as cfg
        from python_agent_harness.client import Client

        stall_started = _threading.Event()

        class StallHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                self.rfile.read(length or b"{}")
                stall_started.set()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                first = b'data: {"choices": [{"delta": {"content": "hi"}}]}\n\n'
                self.wfile.write(f"{len(first):x}".encode() + b"\r\n" + first + b"\r\n")
                self.wfile.flush()
                _time.sleep(60)
                with contextlib.suppress(Exception):
                    self.wfile.write(b"0\r\n\r\n")

            def log_message(self, *a):
                pass

        with tempfile.TemporaryDirectory() as d:
            cfg.SESSION_DIR = Path(d)
            server = ThreadingHTTPServer(("127.0.0.1", 0), StallHandler)
            _threading.Thread(target=server.serve_forever, daemon=True).start()
            host, port = server.server_address
            client = Client(base_url=f"http://{host}:{port}/v1", api_key="test", model="fake")
            out = {}

            def worker():
                try:
                    client.chat([Message(role="user", content="hi")], stream=True)
                    out["r"] = "completed"
                except BaseException as e:  # noqa: BLE001
                    out["r"] = f"{type(e).__name__}: {e}"

            t = _threading.Thread(target=worker, daemon=True)
            t.start()
            try:
                self.assertTrue(stall_started.wait(timeout=5))
                _time.sleep(0.3)  # worker is now blocked in iter_lines
                client.abort()  # what session.cancel() calls on Ctrl-C
                t.join(timeout=5)
                self.assertFalse(t.is_alive(), "worker stuck after abort()")
            finally:
                server.shutdown()
                client.close()
            # the unblocked read surfaces as a network error (the agent
            # loop maps a cancelled state to a clean None, not an error)
            self.assertIn("Error", out["r"])

    def test_cancel_aborts_blocking_chat(self):
        """Ctrl-C during a blocking stream must stop the run, not error."""
        import threading
        import time

        session = RecordingSession()
        session.tools_enabled = True

        class BlockingClient(FakeClient):
            def __init__(self):
                super().__init__([])
                self.unblock = threading.Event()
                self.aborted = False

            def chat(self, *a, **k):
                self.unblock.wait(timeout=10)
                raise RuntimeError("aborted")

            def abort(self):
                self.aborted = True
                self.unblock.set()

        session.client = BlockingClient()
        result = {}
        worker = threading.Thread(
            target=lambda: result.update(
                r=AgentLoop(session, messages=[Message(role="user", content="hi")]).run()
            )
        )
        worker.start()
        time.sleep(0.3)
        session.cancel()
        worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.assertIsNone(result.get("r"))
        self.assertTrue(session.client.aborted)

    def test_cancel_cleared_per_run(self):
        """A new run must not inherit a stale cancel from the previous one."""
        session = RecordingSession()
        session.tools_enabled = False
        session.cancel_event.set()
        session.cancel_event.clear()  # TUI clears before each run
        session.client.script = ["works"]
        loop = AgentLoop(session, messages=[Message(role="user", content="hi")])
        self.assertEqual(loop.run(), "works")

    def test_cancel_mid_tool_round(self):
        session = RecordingSession()
        session.tools_enabled = True
        session.client.script = [
            ("", [ToolCall(id="1", name="Read", arguments='{"file_path": "/tmp/x.py"}')]),
        ]
        session.cancel_event.set()
        loop = AgentLoop(session, messages=[Message(role="user", content="read it")])
        self.assertIsNone(loop.run())

    def test_title_generated_after_loop_finishes(self):
        """The session must get an LLM title once the agent loop completes."""
        session = RecordingSession()
        session.tools_enabled = False
        session.client.script = ["bye"]
        session.client.chat_sync_calls = []

        orig_chat_sync = session.client.chat_sync

        def tracking_chat_sync(
            messages,
            system=None,
            temperature=None,
            max_tokens=None,
            reasoning_effort=None,
            cancel_check=None,
        ):
            session.client.chat_sync_calls.append((messages, system))
            return orig_chat_sync(messages, system=system)

        session.client.chat_sync = tracking_chat_sync
        with mock.patch(
            "python_agent_harness.prompts.read_prompt_file",
            return_value="TITLE-PROMPT",
        ):
            loop = AgentLoop(session, messages=[Message(role="user", content="hi there")])
            loop.run()
        self.assertEqual(len(session.client.chat_sync_calls), 1)
        self.assertEqual(session.client.chat_sync_calls[0][1], "TITLE-PROMPT")
        self.assertEqual(session.store.title, "SYNC-OK")
        self.assertTrue(os.path.basename(session.store.file_path).startswith("SYNC-OK_"))
        # one-shot: a second run must not re-generate the title
        session.client.script = ["again"]
        AgentLoop(session, messages=[Message(role="user", content="hi there")]).run()
        self.assertEqual(len(session.client.chat_sync_calls), 1)

    def test_title_strips_reasoning_and_uses_session_temperature(self):
        """Reasoning content merged by the client must not leak into the
        title (it would become the first 50 chars of the session name),
        and the title request must use the session temperature."""
        session = RecordingSession()
        session.tools_enabled = False
        session.client.script = ["bye"]
        session.client.chat_sync_calls = []

        def chat_sync(
            messages, system=None, temperature=None, max_tokens=None, reasoning_effort=None
        ):
            session.client.chat_sync_calls.append(temperature)
            return (
                Message(
                    role="assistant",
                    content=(
                        "We need to generate a title for the conversation. "
                        "Adding MCP support to agent harness"
                    ),
                    reasoning="We need to generate a title for the conversation.",
                ),
                Usage(),
            )

        session.client.chat_sync = chat_sync
        with mock.patch(
            "python_agent_harness.prompts.read_prompt_file",
            return_value="TITLE-PROMPT",
        ):
            loop = AgentLoop(session, messages=[Message(role="user", content="add mcp")])
            loop.run()
        self.assertEqual(session.store.title, "Adding-MCP-support-to-agent-harness")
        self.assertEqual(session.client.chat_sync_calls, [session.temperature])

    def test_no_title_for_empty_first_message(self):
        session = RecordingSession()
        session.tools_enabled = False
        session.client.script = ["bye"]
        session.client.chat_sync_calls = []
        loop = AgentLoop(session, messages=[Message(role="user", content="")])
        loop.run()
        self.assertEqual(session.client.chat_sync_calls, [])
        self.assertIsNone(session.store.title)

    def test_stale_cancelled_run_does_not_clobber_next_run(self):
        """A cancelled worker finishing late must not overwrite the next
        run's shared history even after the new run cleared the event."""
        session = RecordingSession()
        session.tools_enabled = False

        class AbortClient(FakeClient):
            """Ctrl-C aborts the request; the blocked read raises late,
            after the next run already cleared the shared event."""

            def __init__(self):
                super().__init__([])

            def chat(self, *a, **k):
                session.cancel()  # Ctrl-C: cancel() aborts the HTTP client
                # the next run started meanwhile: `_start_agent` bumps
                # the run generation and clears the shared event
                session.run_generation += 1
                session.cancel_event.clear()
                raise RuntimeError("aborted read")

        session.client = AbortClient()
        # run 1: user asks q1, presses Ctrl-C mid-flight; the stale worker
        # finishes late (after the next run cleared the event)
        loop1 = AgentLoop(session, messages=[Message(role="user", content="q1")])
        # must be treated as a cancel (None), not a spurious error, and
        # must not clobber the shared history with its partial messages
        self.assertIsNone(loop1.run())
        self.assertEqual(session.last_messages, [])
        self.assertIsNone(session.store.title)

        # run 2 completes normally: full history must be present
        session.client = FakeClient(["second answer"])
        loop2 = AgentLoop(session, messages=[Message(role="user", content="q2")])
        self.assertEqual(loop2.run(), "second answer")
        roles = [m.role for m in session.last_messages]
        self.assertEqual(roles, ["user", "assistant"])
        self.assertEqual(session.last_messages[0].text(), "q2")

    def test_cancel_sticks_after_event_cleared(self):
        """A cancelled run stays cancelled once the next run clears the
        shared event (cancel generation + run generation protect state)."""
        session = RecordingSession()
        session.tools_enabled = False
        loop = AgentLoop(session, messages=[Message(role="user", content="q1")])
        loop._cancel_gen = session.cancel_generation  # run() start
        loop._run_gen = session.run_generation
        session.cancel()
        session.run_generation += 1  # next run started
        session.cancel_event.clear()  # ...and cleared the shared event
        self.assertTrue(loop._is_cancelled())
        self.assertTrue(loop._is_stale())

    def test_cancelled_run_keeps_completed_rounds(self):
        """A fully completed tool round survives a cancel that lands in
        a later round: the partial history is cut back to the last
        complete round, so the next turn sends a valid request."""
        session = RecordingSession()
        session.tools_enabled = True
        calls = {"n": 0}
        orig_execute = RecordingSession.execute_tool

        def cancelling_execute(name, args, call_id=None):
            calls["n"] += 1
            if calls["n"] == 2:
                session.cancel()  # Ctrl-C during the second round
            return orig_execute(session, name, args, call_id=call_id)

        session.execute_tool = cancelling_execute
        session.client.script = [
            ("", [ToolCall(id="1", name="Read", arguments='{"file_path": "/tmp/a.py"}')]),
            (
                "",
                [
                    ToolCall(id="2", name="Read", arguments='{"file_path": "/tmp/b.py"}'),
                    ToolCall(id="3", name="Read", arguments='{"file_path": "/tmp/c.py"}'),
                ],
            ),
        ]
        loop = AgentLoop(session, messages=[Message(role="user", content="read all")])
        self.assertIsNone(loop.run())
        # tool 1 (round 1) and tool 2 (which triggered the cancel) ran;
        # tool 3 may or may not have started before the cancel landed
        # (queued tools are skipped after Ctrl-C) — but its result was
        # never delivered to the conversation
        self.assertGreaterEqual(len(session.executed), 2)
        self.assertLessEqual(len(session.executed), 3)
        self.assertFalse(
            any(m.role == "tool" and m.tool_call_id in ("2", "3") for m in loop.messages)
        )
        # the completed round survives; the dangling second round
        # (tool calls 2+3 unanswered) is cut from the shared history
        roles = [m.role for m in session.last_messages]
        self.assertEqual(roles, ["user", "assistant", "tool"])
        self.assertEqual(session.last_messages[-1].text(), "file content")

    def test_cancelled_run_salvages_partial_history(self):
        """Ctrl-C mid-tool-round with no successor must not lose the
        turn: completed content survives, and the dangling round is cut
        so the next turn sends a valid request."""
        session = RecordingSession()
        session.tools_enabled = True
        calls = {"n": 0}
        orig_execute = RecordingSession.execute_tool

        def cancelling_execute(name, args, call_id=None):
            calls["n"] += 1
            if calls["n"] == 2:
                session.cancel()  # Ctrl-C while the 2nd tool runs
            return orig_execute(session, name, args, call_id=call_id)

        session.execute_tool = cancelling_execute
        session.client.script = [
            (
                "",
                [
                    ToolCall(id="1", name="Read", arguments='{"file_path": "/tmp/a.py"}'),
                    ToolCall(id="2", name="Read", arguments='{"file_path": "/tmp/b.py"}'),
                    ToolCall(id="3", name="Read", arguments='{"file_path": "/tmp/c.py"}'),
                ],
            ),
        ]
        loop = AgentLoop(session, messages=[Message(role="user", content="read all")])
        self.assertIsNone(loop.run())
        # tool 1 and tool 2 (which triggered the cancel) ran; tool 3 may
        # or may not have started before the cancel landed — but no
        # results were delivered to the conversation
        self.assertGreaterEqual(len(session.executed), 2)
        self.assertLessEqual(len(session.executed), 3)
        self.assertFalse(any(m.role == "tool" for m in loop.messages))
        # the salvaged history is a valid prefix (user message only —
        # the round's tool results can't stand without the full set)
        self.assertEqual([m.role for m in session.last_messages], ["user"])
        self.assertEqual(session.last_messages[0].text(), "read all")

    def test_cancel_between_chat_and_tools_skips_tools(self):
        """Ctrl-C after the model emitted tool calls but before the tools
        run: the tools must not execute; with no successor the run still
        salvages its (user-only) partial history."""
        session = RecordingSession()
        session.tools_enabled = True

        class CancelAfterChat(FakeClient):
            def __init__(self):
                super().__init__([])

            def chat(self, *a, **k):
                result = super().chat(*a, **k)
                session.cancel()  # Ctrl-C right after the response
                return result

        session.client = CancelAfterChat()
        session.client.script = [
            ("", [ToolCall(id="1", name="Read", arguments='{"file_path": "/tmp/x.py"}')]),
        ]
        loop = AgentLoop(session, messages=[Message(role="user", content="read it")])
        self.assertIsNone(loop.run())
        self.assertEqual(session.executed, [])
        # the assistant tool-call message was dropped with the cancel:
        # only the user message is salvaged
        self.assertEqual([m.text() for m in session.last_messages], ["read it"])

    def test_cancelled_worker_does_not_resurrect_after_clear(self):
        """A cancelled worker winding down after /clear (which bumps the
        run generation WITHOUT starting a new run) must not resurrect
        its salvaged history over the cleared state."""
        session = RecordingSession()
        session.tools_enabled = False
        session.cancel_event.set()
        session.cancel_generation += 1  # Ctrl-C
        loop = AgentLoop(session, messages=[Message(role="user", content="q2")])
        loop._run_gen = session.run_generation  # captured before /clear
        session.last_messages = []  # /clear wiped the shared state
        session.run_generation += 1  # /clear invalidated in-flight workers
        self.assertIsNone(loop.run())
        self.assertEqual(session.last_messages, [])

    def test_compact_and_summary_bump_run_generation(self):
        """/compact and /summary replace the shared conversation, so they
        must invalidate in-flight workers just like /clear and /restore
        (otherwise a dying cancelled worker's salvaged-history commit
        would clobber the compacted/summarized buffer)."""
        session = RecordingSession()
        session.tools_enabled = False
        session.last_messages = [
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi"),
        ]
        gen = session.run_generation
        session.compact_conversation()
        self.assertEqual(session.run_generation, gen + 1)
        self.assertEqual([m.role for m in session.last_messages], ["user", "user"])
        self.assertIn("Compacted Summary", session.last_messages[0].text())
        self.assertEqual(session.last_messages[1].text(), "hello")
        session.summarize_conversation()
        self.assertEqual(session.run_generation, gen + 2)

    def test_stale_worker_does_not_stream_deltas(self):
        """A stale cancelled worker must not stream into the live row."""
        session = RecordingSession()
        session.tools_enabled = False
        deltas = []
        session.on_delta = deltas.append

        class StreamingClient(FakeClient):
            def __init__(self):
                super().__init__([])

            def chat(self, messages, **k):
                session.cancel()  # Ctrl-C while the request is in flight
                on_delta = k.get("on_delta")
                if on_delta:
                    on_delta("partial text")
                return super().chat(messages, **k)

        session.client = StreamingClient()
        session.client.script = ["full answer"]
        loop1 = AgentLoop(session, messages=[Message(role="user", content="q1")])
        self.assertIsNone(loop1.run())
        self.assertEqual(deltas, [])

    def test_compact_no_user_request_returns_false(self):
        """Compaction needs a real (non-nudge) user request to resume
        with; without one it must fail cleanly."""
        session = RecordingSession()
        loop = AgentLoop(session, messages=[Message(role="assistant", content="hi")])
        self.assertFalse(loop.compact())

    def test_compact_keeps_prompts_excluding_nudges_and_frames(self):
        """Compaction preserves every prompt after the summary frame;
        nudges and previous summary frames are dropped (the new summary
        supersedes them), and only the LATEST plan/build reminder batch
        survives — stale read-only plan reminders from before a /plan ->
        /build switch must not contradict the build notice."""
        session = RecordingSession()
        plan_notice = (
            "The plan at /tmp/x/PLAN.md has been approved, you can now edit files. Execute the plan"
        )
        loop = AgentLoop(
            session,
            messages=[
                Message(role="user", content="first"),
                Message(role="assistant", content="ok"),
                Message(
                    role="user",
                    content="<system-reminder>\nPlan mode ACTIVE — READ-ONLY.",
                    injected=True,
                ),
                Message(role="user", content=config.NUDGE_MESSAGE, injected=True),
                Message(role="user", content="second"),
                Message(role="user", content=plan_notice, injected=True),
                Message(
                    role="user",
                    content="**[Compacted Summary]**\n\nold frame\n\n---\n\n"
                    "**[Context compacted]**\n\n---\n\n",
                ),
            ],
        )
        self.assertTrue(loop.compact())
        self.assertEqual([m.role for m in loop.messages], ["user", "user", "user", "user"])
        self.assertTrue(loop.messages[0].text().startswith("**[Compacted Summary]**"))
        self.assertEqual(
            [m.text() for m in loop.messages[1:]],
            ["first", "second", plan_notice],
        )

    def test_compact_empty_summary_returns_false(self):
        """A compaction response with no text must not replace the
        conversation (fail cleanly, reset the compacting flag)."""
        session = RecordingSession()

        def empty_chat_sync(
            messages,
            system=None,
            temperature=None,
            max_tokens=None,
            reasoning_effort=None,
            cancel_check=None,
        ):
            return Message(role="assistant", content=""), Usage()

        session.client.chat_sync = empty_chat_sync
        loop = AgentLoop(session, messages=[Message(role="user", content="do it")])
        self.assertFalse(loop.compact())
        self.assertFalse(session.compacting)

    def test_compact_client_error_returns_false(self):
        """A failing compaction request is non-fatal: it is logged and
        the loop continues without replacing the history."""
        session = RecordingSession()

        def boom_chat_sync(
            messages, system=None, temperature=None, max_tokens=None, reasoning_effort=None
        ):
            raise RuntimeError("compaction API down")

        session.client.chat_sync = boom_chat_sync
        loop = AgentLoop(session, messages=[Message(role="user", content="do it")])
        self.assertFalse(loop.compact())
        self.assertFalse(session.compacting)

    def test_error_state_beats_terminal_response(self):
        """If the loop carries an error state, the error text wins over
        a terminal response (defensive path for the post-supervision
        error check)."""
        session = RecordingSession()
        session.tools_enabled = False
        session.client.script = ["hello"]
        loop = AgentLoop(session, messages=[Message(role="user", content="hi")])
        loop.error = "boom"
        self.assertEqual(loop.run(), "Error: boom")

    def test_zero_budget_subagent_returns_none(self):
        """A sub-agent loop with max_rounds=0 and no assistant text has
        nothing to surface: run() returns None."""
        session = RecordingSession()
        loop = AgentLoop(
            session,
            messages=[Message(role="user", content="hi")],
            top_level=False,
            max_rounds=0,
        )
        self.assertIsNone(loop.run())
