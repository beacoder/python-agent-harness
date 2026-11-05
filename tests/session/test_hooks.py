"""Tests for shell-command hooks (hooks.py) and their wiring into the
tool runner (tool.before/after), the controller (agent.start/end) and
cli.make_session (config loading)."""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import tests  # noqa: F401,E402  (side-effect: hermetic config, see tests/__init__.py)
from python_agent_harness.core.models import Message, ToolCall
from python_agent_harness.core.tool_runner import ToolRunner
from python_agent_harness.entry.controller import Controller
from python_agent_harness.hooks import MAX_TEXT_CHARS, Hook, HookRunner, parse_hooks
from python_agent_harness.prompts.core import user_prompt_texts
from python_agent_harness.session import config
from tests.support import session_sandbox  # noqa: F401  (side-effect: redirect SESSION_DIR)


def reason_of(blocked: str | None, name: str = "Bash") -> str | None:
    """The reason inside a before_tool block message."""
    if blocked is None:
        return None
    prefix = f"Error: tool {name} blocked by hook — "
    assert blocked.startswith(prefix), blocked
    return blocked[len(prefix) :]


def feedback_of(final: str, result: str) -> str | None:
    """The feedback after_tool appended to RESULT."""
    if final == result:
        return None
    prefix = f"{result}\n\n[tool.after hook feedback]\n"
    assert final.startswith(prefix), final
    return final[len(prefix) :]


class HookTestCase(unittest.TestCase):
    """Temp dir for scripts (also the project dir) + a runner factory."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="pah-hooks-")
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.logged: list[str] = []

    def script(self, name: str, body: str) -> str:
        p = Path(self.dir, name)
        p.write_text("#!/bin/sh\n" + body, encoding="utf-8")
        p.chmod(p.stat().st_mode | stat.S_IXUSR)
        return str(p)

    def path(self, name: str) -> Path:
        return Path(self.dir, name)

    def runner(self, section: dict) -> HookRunner:
        return HookRunner(parse_hooks(section), lambda: self.dir, self.logged.append)

    def one(self, event: str, command: str, **entry) -> HookRunner:
        return self.runner({event: [{"command": command, **entry}]})

    def before(self, runner, name="Bash", args=None, is_subagent=False):
        """The block REASON (None = allowed)."""
        return reason_of(
            runner.before_tool(name, args or {"command": "ls"}, "c1", is_subagent), name
        )

    def after(self, runner, result="r"):
        """The FEEDBACK appended to RESULT (None = unchanged)."""
        return feedback_of(runner.after_tool("Edit", {"path": "f"}, "c1", False, result), result)

    @staticmethod
    def wait_agent_end() -> None:
        """agent.end hooks run on their own thread; wait for them."""
        for t in threading.enumerate():
            if t.name == "hook-agent-end":
                t.join(10)


# ---------------------------------------------------------------------------
# config parsing
# ---------------------------------------------------------------------------
class TestParseHooks(unittest.TestCase):
    def test_absent_section(self):
        self.assertEqual(parse_hooks(None), [])

    def test_valid_entries(self):
        hooks = parse_hooks(
            {
                "tool.before": [
                    {"matcher": "Bash", "command": "a", "timeout": 5, "fail_closed": True}
                ],
                "agent.end": [{"command": "b"}],
            }
        )
        self.assertEqual([h.event for h in hooks], ["tool.before", "agent.end"])
        self.assertEqual((hooks[0].timeout, hooks[0].fail_closed), (5.0, True))
        self.assertIsNone(hooks[1].matcher)

    def test_rejects_malformed(self):
        cases = {
            "not an object": [],
            "unknown event": {"tool.whatever": []},
            "entries not a list": {"tool.before": {"command": "x"}},
            "entry not an object": {"tool.before": ["x"]},
            "missing command": {"tool.before": [{}]},
            "blank command": {"tool.before": [{"command": "  "}]},
            "unknown key (typo)": {"tool.before": [{"command": "x", "comand": "y"}]},
            "bad regex": {"tool.before": [{"command": "x", "matcher": "("}]},
            "matcher on agent event": {"agent.end": [{"command": "x", "matcher": "Bash"}]},
            "zero timeout": {"tool.before": [{"command": "x", "timeout": 0}]},
            "bool timeout": {"tool.before": [{"command": "x", "timeout": True}]},
            "fail_closed not bool": {"tool.before": [{"command": "x", "fail_closed": "yes"}]},
            "fail_closed on tool.after": {"tool.after": [{"command": "x", "fail_closed": True}]},
        }
        for label, section in cases.items():
            with self.subTest(label), self.assertRaises(ValueError):
                parse_hooks(section)

    def test_underscore_keys_are_documentation(self):
        hooks = parse_hooks(
            {
                "_comment": "docs",
                "_examples": {"tool.before": [{"command": "x"}]},
                "agent.end": [{"command": "true", "_comment": "per-entry note"}],
            }
        )
        self.assertEqual([h.event for h in hooks], ["agent.end"])

    def test_config_template_hooks_are_inactive_and_examples_valid(self):
        rendered = json.loads(config.CONFIG_TEMPLATE.format(path="/tmp/x.json"))
        section = rendered["hooks"]
        self.assertEqual(parse_hooks(section), [])  # only _comment/_examples
        examples = parse_hooks(section["_examples"])  # each example is valid
        self.assertEqual(
            sorted({h.event for h in examples}),
            ["agent.end", "agent.start", "tool.after", "tool.before"],
        )

    def test_error_locates_entry(self):
        with self.assertRaisesRegex(ValueError, r"hooks\.tool\.before\[1\]"):
            parse_hooks({"tool.before": [{"command": "ok"}, {"command": ""}]})


# ---------------------------------------------------------------------------
# tool.before
# ---------------------------------------------------------------------------
class TestBeforeTool(HookTestCase):
    def test_exit_0_allows(self):
        self.assertIsNone(self.before(self.one("tool.before", "exit 0")))

    def test_exit_2_blocks_with_stderr_reason(self):
        r = self.one("tool.before", "echo 'rm is forbidden' >&2; exit 2")
        self.assertEqual(self.before(r), "rm is forbidden")

    def test_exit_2_without_stderr_names_the_command(self):
        self.assertEqual(self.before(self.one("tool.before", "exit 2")), "blocked by `exit 2`")

    def test_payload_env_and_cwd(self):
        seen, envf = self.path("seen.json"), self.path("env.txt")
        cmd = f'cat > "{seen}"; echo "$PAH_HOOK_EVENT $PAH_PROJECT_DIR $(pwd)" > "{envf}"'
        self.before(self.one("tool.before", cmd), args={"command": "ls"}, is_subagent=True)
        self.assertEqual(
            json.loads(seen.read_text(encoding="utf-8")),
            {
                "event": "tool.before",
                "tool_name": "Bash",
                "tool_args": {"command": "ls"},
                "call_id": "c1",
                "is_subagent": True,
            },
        )
        event, proj, cwd = envf.read_text(encoding="utf-8").split()
        self.assertEqual(event, "tool.before")
        real = os.path.realpath(self.dir)
        self.assertEqual((os.path.realpath(proj), os.path.realpath(cwd)), (real, real))

    def test_failure_is_logged_and_allows(self):
        r = self.one("tool.before", "echo oops >&2; exit 1")
        self.assertIsNone(self.before(r))
        self.assertEqual(
            self.logged, ["tool.before hook `echo oops >&2; exit 1` failed: exited with 1: oops"]
        )

    def test_fail_closed_blocks_on_failure(self):
        r = self.one("tool.before", "exit 1", fail_closed=True)
        reason = self.before(r)
        assert reason is not None
        self.assertIn("failed: exited with 1", reason)
        self.assertTrue(reason.endswith("(fail_closed)"))

    def test_fail_closed_blocks_on_timeout(self):
        reason = self.before(self.one("tool.before", "sleep 5", timeout=0.2, fail_closed=True))
        assert reason is not None
        self.assertIn("timed out after 0.2s", reason)

    def test_matcher_full_match(self):
        ran = self.path("ran")
        r = self.one("tool.before", f'touch "{ran}"; exit 2', matcher="Edit|Write")
        self.assertIsNone(self.before(r, name="Bash"))
        self.assertIsNone(self.before(r, name="EditMac"))  # full match only
        self.assertFalse(ran.exists())
        self.assertIsNotNone(self.before(r, name="Write"))

    def test_first_block_wins(self):
        second_ran = self.path("second")
        r = self.runner(
            {
                "tool.before": [
                    {"command": "echo first >&2; exit 2"},
                    {"command": f'touch "{second_ran}"'},
                ]
            }
        )
        self.assertEqual(self.before(r), "first")
        self.assertFalse(second_ran.exists())

    def test_long_reason_is_clipped(self):
        r = self.one(
            "tool.before", f"head -c {MAX_TEXT_CHARS + 50} /dev/zero | tr '\\0' x >&2; exit 2"
        )
        reason = self.before(r)
        assert reason is not None
        self.assertTrue(reason.endswith("[truncated 50 chars]"))

    def test_timeout_kills_process_group(self):
        marker = self.path("survived")
        r = self.one("tool.before", f'(sleep 1; touch "{marker}") & sleep 5', timeout=0.3)
        self.assertIsNone(self.before(r))
        time.sleep(1.3)
        self.assertFalse(marker.exists())

    def test_detached_child_holding_pipes_does_not_freeze(self):
        child = f"{sys.executable} -c 'import os,time; os.setsid(); time.sleep(4)'"
        r = self.one("tool.before", f"{child} & sleep 30", timeout=0.3)
        start = time.monotonic()
        self.assertIsNone(self.before(r))
        self.assertLess(time.monotonic() - start, 3.5)  # not the child's 4s
        self.assertIn("timed out", self.logged[0])

    def test_unstartable_cwd_falls_back(self):
        r = HookRunner(
            parse_hooks({"tool.before": [{"command": "exit 2"}]}), lambda: "/no/such/dir"
        )
        self.assertEqual(self.before(r), "blocked by `exit 2`")


# ---------------------------------------------------------------------------
# tool.after / agent.start / agent.end
# ---------------------------------------------------------------------------
class TestOtherEvents(HookTestCase):
    def test_after_exit_2_stderr_is_feedback(self):
        self.assertEqual(self.after(self.one("tool.after", "echo E501 >&2; exit 2")), "E501")

    def test_after_exit_0_is_no_feedback(self):
        self.assertIsNone(self.after(self.one("tool.after", "echo formatted 3 files")))

    def test_after_feedback_from_several_hooks_is_joined(self):
        r = self.runner(
            {"tool.after": [{"command": "echo a >&2; exit 2"}, {"command": "echo b >&2; exit 2"}]}
        )
        self.assertEqual(self.after(r), "a\n\nb")

    def test_after_failure_logged(self):
        self.assertIsNone(self.after(self.one("tool.after", "exit 3")))
        self.assertIn("exited with 3", self.logged[0])

    def test_after_payload_has_result(self):
        seen = self.path("seen.json")
        self.after(self.one("tool.after", f'cat > "{seen}"'), result="done")
        payload = json.loads(seen.read_text(encoding="utf-8"))
        self.assertEqual((payload["event"], payload["result"]), ("tool.after", "done"))

    def test_agent_start_stdout_is_context(self):
        r = self.runner(
            {"agent.start": [{"command": "echo 'branch: main'"}, {"command": "echo t-1"}]}
        )
        self.assertEqual(r.agent_start("do it"), "branch: main\n\nt-1")

    def test_agent_start_failure_logged_no_context(self):
        self.assertIsNone(self.one("agent.start", "echo ctx; exit 1").agent_start("x"))
        self.assertEqual(len(self.logged), 1)

    def test_agent_end_payload(self):
        seen = self.path("seen.json")
        thread = self.one("agent.end", f'cat > "{seen}"').agent_end(
            "do it", lambda: ("done", "done")
        )
        assert thread is not None
        thread.join(10)
        self.assertEqual(
            json.loads(seen.read_text(encoding="utf-8")),
            {"event": "agent.end", "text": "do it", "answer": "done", "status": "done"},
        )

    def test_no_hooks_is_a_noop(self):
        r = HookRunner([], lambda: self.dir)
        self.assertIsNone(r.before_tool("Bash", {}, None, False))
        self.assertEqual(r.after_tool("Bash", {}, None, False, "r"), "r")
        self.assertIsNone(r.agent_start("x"))
        self.assertIsNone(r.agent_end("x", lambda: (None, "done")))  # no hooks: no thread

    def test_raising_log_callback_is_contained(self):
        def bad_log(msg):
            raise RuntimeError("sink down")

        r = HookRunner(
            parse_hooks({"agent.end": [{"command": "exit 1"}]}), lambda: self.dir, bad_log
        )
        thread = r.agent_end("x", lambda: (None, "done"))  # must not raise
        assert thread is not None
        thread.join(10)


# ---------------------------------------------------------------------------
# wiring: tool runner
# ---------------------------------------------------------------------------
class FakeToolSession:
    """Just what ToolRunner touches."""

    def __init__(self, hooks):
        self.hooks = hooks
        self.registry = SimpleNamespace(get=lambda name: None)
        self.executed: list[str] = []
        self.last_messages: list = []

    def execute_tool(self, name, args, call_id=None):
        self.executed.append(name)
        return f"result of {name}"

    def notify(self, kind, data=None):
        pass


class TestToolRunnerWiring(HookTestCase):
    def _runner(self, section, top_level=True):
        session = FakeToolSession(self.runner(section))
        loop = SimpleNamespace(
            session=session, top_level=top_level, messages=[], _is_cancelled=lambda: False
        )
        return ToolRunner(loop), session, loop

    def test_blocked_call_never_executes(self):
        runner, session, _ = self._runner({"tool.before": [{"command": "echo no >&2; exit 2"}]})
        result = runner.execute_tool_call(
            ToolCall(id="t1", name="Bash", arguments={"command": "ls"})
        )
        self.assertEqual(result, "Error: tool Bash blocked by hook — no")
        self.assertEqual(session.executed, [])

    def test_allowed_call_executes(self):
        runner, session, _ = self._runner({"tool.before": [{"command": "exit 0"}]})
        self.assertEqual(
            runner.execute_tool_call(ToolCall(id="t1", name="Bash", arguments={})), "result of Bash"
        )
        self.assertEqual(session.executed, ["Bash"])

    def test_subagent_flag(self):
        seen = self.path("seen.json")
        runner, _, _ = self._runner(
            {"tool.before": [{"command": f'cat > "{seen}"'}]}, top_level=False
        )
        runner.execute_tool_call(ToolCall(id="t1", name="Bash", arguments={}))
        self.assertTrue(json.loads(seen.read_text(encoding="utf-8"))["is_subagent"])

    def test_feedback_appended_on_delivery(self):
        runner, _, loop = self._runner(
            {"tool.after": [{"command": "echo 'lint: E501' >&2; exit 2"}]}
        )
        call = ToolCall(id="t1", name="Edit", arguments='{"path": "f"}')
        runner.deliver_tool_result(call, "Error: edit failed")
        (msg,) = loop.messages
        self.assertEqual(
            msg.content, "Error: edit failed\n\n[tool.after hook feedback]\nlint: E501"
        )
        self.assertEqual(call.result, msg.content)

    def test_raw_string_arguments_are_parsed_for_payload(self):
        seen = self.path("seen.json")
        runner, _, _ = self._runner({"tool.after": [{"command": f'cat > "{seen}"'}]})
        runner.deliver_tool_result(ToolCall(id="t1", name="Edit", arguments='{"path": "f"}'), "ok")
        self.assertEqual(json.loads(seen.read_text(encoding="utf-8"))["tool_args"], {"path": "f"})

    def test_no_feedback_leaves_result(self):
        runner, _, loop = self._runner({"tool.after": [{"command": "exit 0"}]})
        runner.deliver_tool_result(ToolCall(id="t1", name="Edit", arguments={}), "ok")
        self.assertEqual(loop.messages[0].content, "ok")

    def test_session_without_hooks_attribute(self):
        session = FakeToolSession(None)
        del session.hooks
        loop = SimpleNamespace(
            session=session, top_level=True, messages=[], _is_cancelled=lambda: False
        )
        runner = ToolRunner(loop)
        self.assertEqual(
            runner.execute_tool_call(ToolCall(id="t1", name="Bash", arguments={})), "result of Bash"
        )
        runner.deliver_tool_result(ToolCall(id="t1", name="Bash", arguments={}), "ok")


# ---------------------------------------------------------------------------
# wiring: controller
# ---------------------------------------------------------------------------
class FakeControllerSession:
    def __init__(self, hooks):
        self.hooks = hooks
        self.project_dir = "/tmp"
        self.run_generation = 1
        self.last_messages: list[Message] = []
        self.cancel_event = threading.Event()
        self.system_prompt = "sys"
        self.notified: list[str] = []

    def notify(self, kind, data=None):
        self.notified.append(kind)

    def log(self, msg):
        pass


class TestControllerWiring(HookTestCase):
    def _run(self, session, side_effect=None):
        captured: list[Message] = []

        def fake_loop(sess, messages, **kw):
            captured.extend(messages)
            if side_effect:
                return side_effect()
            sess.last_messages = [*messages, Message(role="assistant", content="all done")]
            return "all done"

        ctrl = Controller(session)  # type: ignore[arg-type]
        with mock.patch(
            "python_agent_harness.entry.controller.run_agent_loop", side_effect=fake_loop
        ):
            ctrl._run_worker(Message(role="user", content="do it"), seq=1)
        self.wait_agent_end()
        return captured

    def test_context_injected_before_user_message(self):
        session = FakeControllerSession(self.one("agent.start", "echo 'branch: main'"))
        ctx, user = self._run(session)
        self.assertTrue(ctx.injected)
        self.assertEqual(ctx.text(), config.HOOK_CONTEXT_HEADER + "branch: main")
        self.assertEqual((user.text(), user.injected), ("do it", False))

    def test_no_context_no_extra_message(self):
        session = FakeControllerSession(self.one("agent.start", "exit 0"))
        self.assertEqual([m.text() for m in self._run(session)], ["do it"])

    def test_agent_end_gets_answer(self):
        seen = self.path("seen.json")
        session = FakeControllerSession(self.one("agent.end", f'cat > "{seen}"'))
        self._run(session)
        self.assertEqual(json.loads(seen.read_text(encoding="utf-8"))["answer"], "all done")
        self.assertIn("run_finished", session.notified)

    def test_agent_end_fires_for_superseded_run(self):
        seen = self.path("seen.json")
        session = FakeControllerSession(self.one("agent.end", f'cat > "{seen}"'))

        def supersede():
            session.run_generation += 1
            return "stale"

        self._run(session, side_effect=supersede)
        self.assertEqual(json.loads(seen.read_text(encoding="utf-8"))["answer"], "stale")

    def test_agent_end_fires_when_the_loop_crashes(self):
        seen = self.path("seen.json")
        session = FakeControllerSession(self.one("agent.end", f'cat > "{seen}"'))

        def crash():
            raise RuntimeError("kaput")

        self._run(session, side_effect=crash)
        self.assertIsNone(json.loads(seen.read_text(encoding="utf-8"))["answer"])
        self.assertIn("run_finished", session.notified)

    def test_context_excluded_from_title_and_compaction(self):
        """Content match: restored sessions lose the ``injected`` flag."""
        from python_agent_harness.session.session import Session
        from tests.support.agent_test_utils import FakeClient

        ctx = Message(role="user", content=config.HOOK_CONTEXT_HEADER + "branch: main")
        user = Message(role="user", content="do it")
        self.assertEqual(user_prompt_texts([ctx, user]), ["do it"])
        session = Session(
            project_dir=self.dir, client=FakeClient([]), model="gpt-5-mini", model_profiles={}
        )
        try:
            session.remember_user_text([user, ctx])
            self.assertEqual(session.store.first_user_message(), "do it")
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Ctrl-C (cancel) and long-running commands
# ---------------------------------------------------------------------------
class TestCancel(HookTestCase):
    def cancellable(self, section):
        flag = threading.Event()
        runner = HookRunner(parse_hooks(section), lambda: self.dir, self.logged.append, flag.is_set)
        return runner, flag

    def _cancel_after(self, flag, delay):
        threading.Timer(delay, flag.set).start()

    def test_cancel_kills_running_before_hook_and_blocks(self):
        runner, flag = self.cancellable({"tool.before": [{"command": "sleep 10"}]})
        self._cancel_after(flag, 0.3)
        start = time.monotonic()
        reason = reason_of(runner.before_tool("Bash", {}, None, False))
        self.assertLess(time.monotonic() - start, 2)
        self.assertEqual(reason, "cancelled by the user")
        self.assertEqual(self.logged, [])  # a cancel is not a hook failure

    def test_already_cancelled_skips_spawning(self):
        ran = self.path("ran")
        runner, flag = self.cancellable({"tool.before": [{"command": f'touch "{ran}"'}]})
        flag.set()
        self.assertEqual(
            reason_of(runner.before_tool("Bash", {}, None, False)), "cancelled by the user"
        )
        time.sleep(0.2)
        self.assertFalse(ran.exists())

    def test_cancel_interrupts_after_and_start(self):
        runner, flag = self.cancellable(
            {
                "tool.after": [{"command": "sleep 10"}],
                "agent.start": [{"command": "sleep 10"}],
            }
        )
        self._cancel_after(flag, 0.2)
        start = time.monotonic()
        self.assertEqual(runner.after_tool("Bash", {}, None, False, "r"), "r")
        self.assertIsNone(runner.agent_start("x"))  # flag already set: no spawn
        self.assertLess(time.monotonic() - start, 2)

    def test_agent_end_runs_despite_cancel(self):
        seen = self.path("seen.json")
        runner, flag = self.cancellable({"agent.end": [{"command": f'sleep 0.3; cat > "{seen}"'}]})
        flag.set()
        thread = runner.agent_end("x", lambda: (None, "cancelled"))
        assert thread is not None
        thread.join(10)
        self.assertEqual(json.loads(seen.read_text(encoding="utf-8"))["status"], "cancelled")

    def test_large_payload_with_slow_reader_arrives_whole(self):
        """The payload outgrows the pipe buffer and the reader starts
        late, so writing spans several poll slices."""
        big = "x" * 300_000
        r = self.one("tool.before", "sleep 0.4; wc -c >&2; exit 2")
        reason = reason_of(r.before_tool("Write", {"content": big}, None, False), "Write")
        payload_len = len(
            json.dumps(
                {
                    "event": "tool.before",
                    "tool_name": "Write",
                    "tool_args": {"content": big},
                    "call_id": None,
                    "is_subagent": False,
                }
            )
        )
        self.assertEqual(int(reason or 0), payload_len)


# ---------------------------------------------------------------------------
# agent.end status (controller)
# ---------------------------------------------------------------------------
class TestRunStatus(HookTestCase):
    def status_of(self, side_effect=None, cancel=False):
        seen = self.path("status.json")
        session = FakeControllerSession(self.one("agent.end", f'cat > "{seen}"'))
        if cancel:
            session.cancel_event.set()
        ctrl = Controller(session)  # type: ignore[arg-type]
        with mock.patch(
            "python_agent_harness.entry.controller.run_agent_loop",
            side_effect=lambda *a, **k: side_effect(session) if side_effect else "ok",
        ):
            ctrl._run_worker(Message(role="user", content="x"), seq=1)
        self.wait_agent_end()
        return json.loads(seen.read_text(encoding="utf-8"))["status"]

    def test_done(self):
        self.assertEqual(self.status_of(), "done")

    def test_error_result_text(self):
        self.assertEqual(self.status_of(lambda s: "Error: 429 rate limited"), "error")

    def test_crash(self):
        def crash(s):
            raise RuntimeError("kaput")

        self.assertEqual(self.status_of(crash), "error")

    def test_cancelled(self):
        self.assertEqual(self.status_of(lambda s: None, cancel=True), "cancelled")

    def test_superseded_beats_cancelled(self):
        def supersede(s):
            s.run_generation += 1
            return None

        self.assertEqual(self.status_of(supersede, cancel=True), "superseded")


# ---------------------------------------------------------------------------
# agent.end does not block the run; its answer is the shown answer
# ---------------------------------------------------------------------------
class TestAgentEndAsync(HookTestCase):
    def test_returns_at_once_on_a_non_daemon_thread(self):
        done = self.path("done")
        r = self.one("agent.end", f'sleep 0.8; touch "{done}"')
        start = time.monotonic()
        thread = r.agent_end("x", lambda: ("a", "done"))
        self.assertLess(time.monotonic() - start, 0.5)
        assert thread is not None
        self.assertFalse(thread.daemon)  # a headless exit still waits for it
        thread.join(10)
        self.assertTrue(done.exists())

    def test_slow_agent_end_does_not_delay_run_finished(self):
        session = FakeControllerSession(self.one("agent.end", "sleep 1"))
        ctrl = Controller(session)  # type: ignore[arg-type]
        start = time.monotonic()
        with mock.patch("python_agent_harness.entry.controller.run_agent_loop", return_value="ok"):
            ctrl._run_worker(Message(role="user", content="x"), seq=1)
        self.assertLess(time.monotonic() - start, 0.7)
        self.assertIn("run_finished", session.notified)
        self.wait_agent_end()

    def _answer(self, messages_after_prompt, raw):
        seen = self.path("seen.json")
        session = FakeControllerSession(self.one("agent.end", f'cat > "{seen}"'))

        def fake_loop(sess, messages, **kw):
            sess.last_messages = [*messages, *messages_after_prompt]
            return raw

        ctrl = Controller(session)  # type: ignore[arg-type]
        with mock.patch(
            "python_agent_harness.entry.controller.run_agent_loop", side_effect=fake_loop
        ):
            ctrl._run_worker(Message(role="user", content="q"), seq=1)
        self.wait_agent_end()
        return json.loads(seen.read_text(encoding="utf-8"))

    def test_answer_skips_reasoning_and_check_only_message(self):
        check = "[FINAL CHECK]\n- Goal: g\n- Status: SUCCESS\n- Evidence: e"
        payload = self._answer(
            [
                Message(
                    role="assistant",
                    content="Thinking hard. The answer is 4.",
                    reasoning="Thinking hard. ",
                ),
                Message(role="assistant", content=check),
            ],
            raw=check,  # what the loop returns for a supervised run
        )
        self.assertEqual((payload["status"], payload["answer"]), ("done", "The answer is 4."))

    def test_error_run_reports_error_text(self):
        payload = self._answer(
            [Message(role="assistant", content="partial progress")],
            raw="Error: 429 rate limited",
        )
        self.assertEqual(
            (payload["status"], payload["answer"]), ("error", "Error: 429 rate limited")
        )

    def test_no_answer_is_null(self):
        payload = self._answer([], raw=None)
        self.assertIsNone(payload["answer"])


# ---------------------------------------------------------------------------
# SIGTERM before SIGKILL
# ---------------------------------------------------------------------------
class TestGracefulStop(HookTestCase):
    def test_timeout_sends_sigterm_first(self):
        marker = self.path("cleaned-up")
        cmd = f"trap 'touch \"{marker}\"; exit 0' TERM; sleep 10 & wait"
        self.one("tool.before", cmd, timeout=0.3).before_tool("Bash", {}, None, False)
        self.assertTrue(marker.exists())

    def test_cancel_sends_sigterm_first(self):
        marker = self.path("cleaned-up")
        flag = threading.Event()
        cmd = f"trap 'touch \"{marker}\"; exit 0' TERM; sleep 10 & wait"
        r = HookRunner(
            parse_hooks({"tool.before": [{"command": cmd}]}), lambda: self.dir, None, flag.is_set
        )
        threading.Timer(0.3, flag.set).start()
        self.assertEqual(reason_of(r.before_tool("Bash", {}, None, False)), "cancelled by the user")
        self.assertTrue(marker.exists())

    def test_term_ignoring_hook_is_still_killed(self):
        # SIGTERM is ignored by the shell AND inherited as ignored by sleep
        r = self.one("tool.before", "trap '' TERM; sleep 10", timeout=0.3)
        start = time.monotonic()
        self.assertIsNone(r.before_tool("Bash", {}, None, False))
        self.assertLess(time.monotonic() - start, 3)
        self.assertIn("timed out", self.logged[0])

    def test_term_ignoring_hook_is_still_killed_on_cancel(self):
        flag = threading.Event()
        r = HookRunner(
            parse_hooks({"tool.before": [{"command": "trap '' TERM; sleep 10"}]}),
            lambda: self.dir,
            None,
            flag.is_set,
        )
        threading.Timer(0.2, flag.set).start()
        start = time.monotonic()
        self.assertEqual(reason_of(r.before_tool("Bash", {}, None, False)), "cancelled by the user")
        self.assertLess(time.monotonic() - start, 3)


# ---------------------------------------------------------------------------
# the containment boundary (HookRunner._guard)
# ---------------------------------------------------------------------------
class TestBoundary(HookTestCase):
    """Every public method survives an internal error with its default."""

    def boom(self, *a, **k):
        raise RuntimeError("internal bug")

    def test_before_tool_internal_error_allows(self):
        r = self.one("tool.before", "exit 2")
        with mock.patch.object(r, "_before_tool", self.boom):
            self.assertIsNone(r.before_tool("Bash", {}, None, False))
        self.assertIn(
            "tool.before hooks: internal error (RuntimeError: internal bug)", self.logged[0]
        )

    def test_before_tool_internal_error_blocks_when_fail_closed(self):
        r = self.one("tool.before", "exit 0", fail_closed=True)
        with mock.patch.object(r, "_before_tool", self.boom):
            blocked = r.before_tool("Bash", {}, None, False)
        self.assertEqual(
            blocked, "Error: tool Bash blocked by hook — internal hook error (fail_closed)"
        )

    def test_after_tool_internal_error_keeps_result(self):
        r = self.one("tool.after", "exit 2")
        with mock.patch.object(r, "_after_tool", self.boom):
            self.assertEqual(r.after_tool("Edit", {}, None, False, "the result"), "the result")

    def test_agent_start_internal_error_no_context(self):
        r = self.one("agent.start", "echo ctx")
        with mock.patch.object(r, "_agent_start", self.boom):
            self.assertIsNone(r.agent_start("x"))

    def test_agent_end_outcome_error(self):
        r = self.one("agent.end", "true")
        self.assertIsNone(r.agent_end("x", self.boom))
        self.assertIn("agent.end hooks: internal error", self.logged[0])

    def test_agent_end_thread_start_failure(self):
        r = self.one("agent.end", "true")
        with mock.patch.object(
            threading.Thread, "start", side_effect=RuntimeError("can't start new thread")
        ):
            self.assertIsNone(r.agent_end("x", lambda: (None, "done")))
        self.assertIn("can't start new thread", self.logged[0])

    def test_agent_end_thread_body_error_does_not_escape(self):
        escaped: list = []
        r = self.one("agent.end", "true")
        with (
            mock.patch.object(threading, "excepthook", lambda args: escaped.append(args)),
            mock.patch.object(r, "_fire", self.boom),
        ):
            thread = r.agent_end("x", lambda: (None, "done"))
            assert thread is not None
            thread.join(10)
        self.assertEqual(escaped, [])
        self.assertIn("agent.end hooks: internal error", self.logged[0])

    def test_pathologically_nested_arguments(self):
        seen = self.path("seen.json")
        r = self.one("tool.after", f'cat > "{seen}"')
        deep = "[" * 100_000 + "]" * 100_000
        self.assertEqual(r.after_tool("Edit", deep, None, False, "ok"), "ok")
        # json.loads raised RecursionError: the hook still ran, with the raw string
        self.assertEqual(json.loads(seen.read_text(encoding="utf-8"))["tool_args"], deep)

    def test_unserializable_payload_is_contained(self):
        r = self.one("tool.before", "exit 2")
        with mock.patch("python_agent_harness.hooks.json.dumps", self.boom):
            # json.dumps fails inside _run -> a per-hook failure outcome,
            # not an exception: logged, call allowed (not fail_closed)
            self.assertIsNone(r.before_tool("Bash", {}, None, False))
        self.assertIn("internal bug", self.logged[0])

    def test_tool_runner_delivery_survives_internal_error(self):
        session = FakeToolSession(self.one("tool.after", "exit 2"))
        loop = SimpleNamespace(
            session=session, top_level=True, messages=[], _is_cancelled=lambda: False
        )
        with mock.patch.object(session.hooks, "_after_tool", self.boom):
            ToolRunner(loop).deliver_tool_result(ToolCall(id="t1", name="Edit", arguments={}), "ok")
        self.assertEqual(loop.messages[0].content, "ok")

    def test_controller_finishes_run_when_end_outcome_fails(self):
        session = FakeControllerSession(self.one("agent.end", "true"))
        ctrl = Controller(session)  # type: ignore[arg-type]
        with (
            mock.patch("python_agent_harness.entry.controller.run_agent_loop", return_value="ok"),
            mock.patch.object(ctrl, "_end_outcome", self.boom),
        ):
            ctrl._run_worker(Message(role="user", content="x"), seq=1)
        self.assertIn("run_finished", session.notified)


# ---------------------------------------------------------------------------
# resource bounds and secrets
# ---------------------------------------------------------------------------
class TestResourceBounds(HookTestCase):
    def test_runaway_output_is_bounded(self):
        """`yes` writes as fast as it can until the timeout: memory stays
        bounded and the hook still ends at its timeout."""
        from python_agent_harness.hooks import MAX_CAPTURE_BYTES, _run

        (hook,) = parse_hooks({"tool.after": [{"command": "yes", "timeout": 1}]})
        start = time.monotonic()
        outcome = _run(hook, {}, self.dir, dict(os.environ), None)
        self.assertLess(time.monotonic() - start, 4)
        self.assertEqual(outcome.error, "timed out after 1s")
        # nothing kept beyond the cap even though gigabytes were written

        (hook,) = parse_hooks({"tool.after": [{"command": "yes | head -c 5000000; exit 2"}]})
        outcome = _run(hook, {}, self.dir, dict(os.environ), None)
        self.assertEqual(outcome.code, 2)
        self.assertLessEqual(len(outcome.stdout.encode()), MAX_CAPTURE_BYTES + 100)
        self.assertTrue(outcome.stdout.endswith("[output beyond the capture limit discarded]"))

    def test_big_stderr_feedback_is_clipped_for_the_model(self):
        r = self.one("tool.after", "yes E501 | head -c 1000000 >&2; exit 2")
        feedback = self.after(r)
        assert feedback is not None
        self.assertLess(len(feedback), MAX_TEXT_CHARS + 200)

    def test_background_holder_does_not_delay_a_finished_hook(self):
        # the hook exits at once, but its background job holds stdout 5s
        r = self.one("tool.before", "(sleep 5) & exit 0")
        start = time.monotonic()
        self.assertIsNone(self.before(r))
        self.assertLess(time.monotonic() - start, 2.5)

    def test_hook_not_reading_stdin_with_big_payload(self):
        r = self.one("tool.before", "exit 2")  # never reads stdin
        start = time.monotonic()
        self.assertEqual(self.before(r, args={"content": "x" * 1_000_000}), "blocked by `exit 2`")
        self.assertLess(time.monotonic() - start, 3)

    def test_nul_in_command_cannot_start(self):
        r = HookRunner([Hook("tool.before", "a\0b")], lambda: self.dir, self.logged.append)
        self.assertIsNone(self.before(r))
        self.assertIn("cannot start", self.logged[0])


class TestSecrets(HookTestCase):
    def _env_seen(self, entry):
        out = self.path("env.txt")
        r = self.runner({"tool.before": [{"command": f'env > "{out}"', **entry}]})
        with mock.patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "sk-secret", "SLACK_TOKEN": "xoxb-secret", "HOOKTEST_PLAIN": "ok"},
        ):
            r.before_tool("Bash", {}, None, False)
        return out.read_text(encoding="utf-8")

    def test_secret_looking_vars_are_scrubbed(self):
        env = self._env_seen({})
        self.assertNotIn("sk-secret", env)
        self.assertNotIn("xoxb-secret", env)
        self.assertIn("HOOKTEST_PLAIN=ok", env)
        self.assertIn("PAH_HOOK_EVENT=tool.before", env)

    def test_env_passthrough_is_per_hook(self):
        env = self._env_seen({"env_passthrough": ["SLACK_TOKEN"]})
        self.assertIn("SLACK_TOKEN=xoxb-secret", env)
        self.assertNotIn("sk-secret", env)

    def test_env_passthrough_validated(self):
        for bad in ("SLACK_TOKEN", [1], [""]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_hooks({"tool.before": [{"command": "x", "env_passthrough": bad}]})


# ---------------------------------------------------------------------------
# review fixes: finite timeouts, spawn order, fail_closed fallback scope,
# capture flag, post-exit drain, env built once; status + config listing
# ---------------------------------------------------------------------------
class TestReviewFixes(HookTestCase):
    def test_nan_and_infinite_timeouts_rejected(self):
        for raw in ("NaN", "Infinity", "-Infinity"):
            section = json.loads(f'{{"tool.before": [{{"command": "x", "timeout": {raw}}}]}}')
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, "positive number"):
                parse_hooks(section)

    def test_unserializable_payload_spawns_nothing(self):
        from python_agent_harness.hooks import _run

        (hook,) = parse_hooks({"tool.before": [{"command": "true"}]})
        circular: dict = {}
        circular["self"] = circular
        with mock.patch("python_agent_harness.hooks.subprocess.Popen") as popen:
            outcome = _run(hook, {"tool_args": circular}, self.dir, {}, None)
        popen.assert_not_called()
        assert outcome.error is not None
        self.assertIn("payload not serializable", outcome.error)

    def test_fail_closed_fallback_respects_matcher(self):
        r = self.one("tool.before", "exit 0", matcher="Bash", fail_closed=True)
        with mock.patch.object(r, "_before_tool", side_effect=RuntimeError("bug")):
            self.assertIsNone(r.before_tool("Read", {}, None, False))  # not covered
            self.assertIsNotNone(r.before_tool("Bash", {}, None, False))  # covered

    def test_capture_flag_only_when_bytes_were_dropped(self):
        from python_agent_harness.hooks import MAX_CAPTURE_BYTES, _run

        for size, dropped in ((MAX_CAPTURE_BYTES, False), (MAX_CAPTURE_BYTES + 1, True)):
            (hook,) = parse_hooks({"tool.before": [{"command": f"head -c {size} /dev/zero"}]})
            outcome = _run(hook, {}, self.dir, dict(os.environ), None)
            with self.subTest(size=size):
                self.assertEqual(outcome.stdout.endswith("limit discarded]"), dropped)

    def test_background_holder_costs_a_short_drain_only(self):
        r = self.one("tool.before", "(sleep 5) & exit 0")
        start = time.monotonic()
        self.assertIsNone(self.before(r))
        self.assertLess(time.monotonic() - start, 0.8)  # drain 0.25s, not 1s

    def test_env_scrubbed_once_per_event(self):
        r = self.runner(
            {"tool.before": [{"command": "true"}, {"command": "true"}, {"command": "true"}]}
        )
        with mock.patch(
            "python_agent_harness.tools.bash.scrubbed_env", side_effect=lambda env: dict(env)
        ) as scrub:
            r.before_tool("Bash", {}, None, False)
        self.assertEqual(scrub.call_count, 1)

    def test_status_announces_running_hooks(self):
        shown: list = []
        r = HookRunner(
            parse_hooks(
                {
                    "tool.before": [{"command": "true"}],
                    "agent.end": [{"command": "true"}],
                }
            ),
            lambda: self.dir,
            None,
            None,
            shown.append,
        )
        r.before_tool("Bash", {}, None, False)
        self.assertEqual(shown, ["hook tool.before: true", None])
        thread = r.agent_end("x", lambda: (None, "done"))
        assert thread is not None
        thread.join(10)
        self.assertEqual(len(shown), 2)  # agent.end runs after the run: no status

    def test_label_shows_the_program_not_the_arguments(self):
        from python_agent_harness.hooks import Hook, _label

        label = _label(Hook("tool.before", "curl -H 'Authorization: Bearer s3cret' https://x"))
        self.assertEqual(label, "hook tool.before: curl")
        self.assertNotIn("s3cret", label)
        self.assertEqual(
            _label(Hook("tool.before", "~/bin/guard.sh --flag argument")),
            "hook tool.before: guard.sh",
        )
        self.assertEqual(
            _label(Hook("tool.before", "FOO=1 /usr/bin/ruff   check -q .")),
            "hook tool.before: ruff",
        )
        self.assertEqual(_label(Hook("tool.before", '"~/bin/q.sh" --x')), "hook tool.before: q.sh")
        # a quoted program containing spaces must not be split on them
        self.assertEqual(
            _label(Hook("tool.before", '"/opt/my dir/guard.sh" --flag')),
            "hook tool.before: guard.sh",
        )

    def test_label_tolerates_empty_quotes_and_unbalanced_quoting(self):
        """An empty quoted word must not raise or become the program."""
        from python_agent_harness.hooks import Hook, _label

        for command in ("''", '""', "'' --x", "A=1 ''", "'   '", '"a', "guard.sh' --a", 'x"y'):
            label = _label(Hook("tool.before", command))  # must not raise
            self.assertTrue(label.startswith("hook tool.before"), label)
        self.assertEqual(_label(Hook("tool.before", '"" guard.sh')), "hook tool.before: guard.sh")
        self.assertEqual(
            _label(Hook("tool.before", 'curl -H "Bearer s3cret')), "hook tool.before: curl"
        )

    def test_label_never_falls_back_to_the_raw_command(self):
        """A command that is only assignments (or blank) has no program to show."""
        from python_agent_harness.hooks import Hook, _label

        label = _label(Hook("agent.start", "TOKEN=s3cret"))
        self.assertEqual(label, "hook agent.start")
        self.assertNotIn("s3cret", label)
        self.assertEqual(_label(Hook("tool.before", "   ")), "hook tool.before")

    def test_session_wires_status_to_hook_running_notify(self):
        from python_agent_harness.session.session import Session
        from tests.support.agent_test_utils import FakeClient

        session = Session(
            project_dir=self.dir, client=FakeClient([]), model="gpt-5-mini", model_profiles={}
        )
        try:
            seen: list = []
            session.notify_fn = lambda kind, data=None: seen.append((kind, data))
            session.install_hooks(parse_hooks({"tool.before": [{"command": "true"}]}))
            session.hooks.before_tool("Bash", {}, None, False)
            self.assertEqual(
                seen, [("hook_running", "hook tool.before: true"), ("hook_running", None)]
            )
        finally:
            session.close()

    def test_tui_shows_then_restores_status(self):
        from tests.support.tui_test_utils import make_tui

        tui, _ = make_tui()
        tui.status = " ⏳ Bash"
        tui._on_notify("hook_running", "hook tool.before: a")
        tui._on_notify("hook_running", "hook tool.before: b")  # parallel round
        self.assertEqual(tui.status, " ⏳ hook tool.before: b")
        tui._on_notify("hook_running", None)
        self.assertEqual(tui.status, " ⏳ hook tool.before: b")  # one still running
        tui._on_notify("hook_running", None)
        self.assertEqual(tui.status, " ⏳ Bash")
        tui._on_notify("hook_running", None)  # unmatched end: ignored
        self.assertEqual(tui.status, " ⏳ Bash")

    def test_config_command_lists_hooks(self):
        import argparse
        import contextlib
        import io

        from python_agent_harness.entry import cli

        cfg = self.path("config.json")
        cfg.write_text(
            json.dumps(
                {
                    "hooks": {
                        "tool.before": [{"matcher": "Bash", "command": "g.sh", "fail_closed": True}]
                    }
                }
            ),
            encoding="utf-8",
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli.cmd_config(argparse.Namespace(path=str(cfg), init=False))
        self.assertIn("  tool.before [Bash]: g.sh (timeout=30s, fail_closed)", buf.getvalue())

        cfg.write_text(json.dumps({"hooks": {"tool.before": [{}]}}), encoding="utf-8")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli.cmd_config(argparse.Namespace(path=str(cfg), init=False))
        self.assertIn("hooks: INVALID", buf.getvalue())


# ---------------------------------------------------------------------------
# call sites tolerate a replaced / mock ``hooks`` object
# ---------------------------------------------------------------------------
class TestForeignHooksObject(HookTestCase):
    def test_tool_runner_ignores_non_string_returns(self):
        session = FakeToolSession(mock.Mock())  # every method returns a Mock
        loop = SimpleNamespace(
            session=session, top_level=True, messages=[], _is_cancelled=lambda: False
        )
        runner = ToolRunner(loop)
        result = runner.execute_tool_call(ToolCall(id="t1", name="Bash", arguments={}))
        self.assertEqual(result, "result of Bash")  # not blocked by a Mock
        runner.deliver_tool_result(ToolCall(id="t1", name="Bash", arguments={}), "ok")
        self.assertEqual(loop.messages[0].content, "ok")  # not replaced by a Mock

    def test_controller_ignores_non_string_context(self):
        session = FakeControllerSession(mock.Mock())
        ctrl = Controller(session)  # type: ignore[arg-type]
        with mock.patch(
            "python_agent_harness.entry.controller.run_agent_loop", return_value="ok"
        ) as loop:
            ctrl._run_worker(Message(role="user", content="hi"), seq=1)
        loop.assert_called_once()
        self.assertEqual([m.text() for m in loop.call_args.kwargs["messages"]], ["hi"])
        self.assertIn("run_finished", session.notified)

    def test_tui_restores_latest_tool_status_after_hook(self):
        from tests.support.tui_test_utils import make_tui

        tui, _ = make_tui()
        tui._on_notify("tool_running", "Read")
        tui._on_notify("hook_running", "hook tool.before: g")
        tui._on_notify("tool_running", "Grep")  # another pool thread, mid-hook
        self.assertEqual(tui.status, " ⏳ hook tool.before: g")  # hook stays visible
        tui._on_notify("hook_running", None)
        self.assertEqual(tui.status, " ⏳ Grep")  # not the stale " ⏳ Read"


# ---------------------------------------------------------------------------
# config loading (cli.make_session)
# ---------------------------------------------------------------------------
class TestConfigLoading(HookTestCase):
    ENV_KEYS = ["OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"] + [
        "OPENAI_SUBAGENT_BASE_URL",
        "OPENAI_SUBAGENT_API_KEY",
        "OPENAI_SUBAGENT_MODEL",
    ]

    def setUp(self):
        super().setUp()
        for k in self.ENV_KEYS:
            saved = os.environ.pop(k, None)
            if saved is not None:
                self.addCleanup(os.environ.__setitem__, k, saved)
        cfg = tempfile.TemporaryDirectory()
        self.addCleanup(cfg.cleanup)
        self.config_path = str(Path(cfg.name, "config.json"))

    def write(self, data):
        Path(self.config_path).write_text(json.dumps(data), encoding="utf-8")

    def test_hooks_installed_on_session(self):
        from python_agent_harness.entry import cli

        self.write(
            {"hooks": {"tool.before": [{"command": "exit 2"}], "agent.end": [{"command": "true"}]}}
        )
        session = cli.make_session(self.dir, config_path=self.config_path)
        try:
            self.assertEqual(len(session.hooks), 2)
            self.assertEqual(
                reason_of(session.hooks.before_tool("Bash", {}, None, False)), "blocked by `exit 2`"
            )
        finally:
            session.close()

    def test_no_section_means_no_hooks(self):
        from python_agent_harness.entry import cli

        self.write({})
        session = cli.make_session(self.dir, config_path=self.config_path)
        try:
            self.assertEqual(len(session.hooks), 0)
        finally:
            session.close()

    def test_invalid_section_stops_startup_before_building_session(self):
        from python_agent_harness.entry import cli

        self.write({"hooks": {"tool.before": [{"command": ""}]}})
        with (
            mock.patch("python_agent_harness.entry.cli.Session") as session_cls,
            self.assertRaisesRegex(
                ValueError, r"config file .*config\.json: hooks\.tool\.before\[0\]"
            ),
        ):
            cli.make_session(self.dir, config_path=self.config_path)
        session_cls.assert_not_called()

    def test_broken_file_raises(self):
        Path(self.config_path).write_text("{broken", encoding="utf-8")
        with self.assertRaises(ValueError):
            config.load_hooks(self.config_path)


if __name__ == "__main__":
    unittest.main()
