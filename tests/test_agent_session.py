"""AgentSession unit tests: dir discovery, plan-mode guards, skills,
auto-save/title/cancel edge cases, compact/summarize failure paths."""

import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))

from python_agent_harness import config
from python_agent_harness.agent_session import (
    AgentSession,
    find_context_dir,
    find_skill_dir,
)
from python_agent_harness.models import Message, Usage

from test_agent import FakeClient, RecordingSession


class TestFindDirs(unittest.TestCase):
    """find_skill_dir / find_context_dir resolution: configured path
    wins, then default locations, then None."""

    def test_skill_dir_configured_wins(self):
        with tempfile.TemporaryDirectory(prefix="pah-skills-") as d:
            self.assertEqual(find_skill_dir("/proj", d), d)

    def test_skill_dir_default_project_fallback(self):
        """A configured-but-missing path falls through to the project's
        skills/ directory (first existing default wins)."""
        with tempfile.TemporaryDirectory(prefix="pah-proj-") as proj:
            skills = os.path.join(proj, "skills")
            os.makedirs(skills)
            with mock.patch(
                "python_agent_harness.agent_session.os.path.expanduser",
                return_value=os.path.join(proj, "no-home"),
            ):
                self.assertEqual(find_skill_dir(proj, "/nonexistent"), skills)

    def test_skill_dir_default_home_fallback(self):
        with tempfile.TemporaryDirectory(prefix="pah-home-") as home:
            emacs = os.path.join(home, ".emacs.d", "skills")
            os.makedirs(emacs)
            with mock.patch(
                "python_agent_harness.agent_session.os.path.expanduser",
                return_value=home,
            ):
                self.assertEqual(find_skill_dir("/proj", None), emacs)

    def test_skill_dir_none_when_missing(self):
        with tempfile.TemporaryDirectory(prefix="pah-proj-") as proj:
            with mock.patch(
                "python_agent_harness.agent_session.os.path.expanduser",
                return_value=os.path.join(proj, "no-home"),
            ):
                self.assertIsNone(find_skill_dir(proj, None))

    def test_context_dir_configured_wins(self):
        with tempfile.TemporaryDirectory(prefix="pah-ctx-") as d:
            self.assertEqual(find_context_dir("/proj", d), d)

    def test_context_dir_project_fallback(self):
        with tempfile.TemporaryDirectory(prefix="pah-proj-") as proj:
            ctx = os.path.join(proj, "contexts")
            os.makedirs(ctx)
            with mock.patch(
                "python_agent_harness.agent_session.os.path.expanduser",
                return_value=os.path.join(proj, "no-home"),
            ):
                self.assertEqual(find_context_dir(proj, None), ctx)

    def test_context_dir_none_when_missing(self):
        with tempfile.TemporaryDirectory(prefix="pah-proj-") as proj:
            with mock.patch(
                "python_agent_harness.agent_session.os.path.expanduser",
                return_value=os.path.join(proj, "no-home"),
            ):
                self.assertIsNone(find_context_dir(proj, None))


class TestSessionInteractive(unittest.TestCase):
    """confirm / ask_questions without TUI hooks fall back to safe
    defaults (confirm=True, "Unanswered")."""

    def test_confirm_defaults_true_without_hook(self):
        session = RecordingSession()
        self.assertTrue(session.confirm("proceed?"))

    def test_confirm_uses_hook_when_set(self):
        session = RecordingSession()
        session.confirm_fn = lambda prompt: False
        self.assertFalse(session.confirm("proceed?"))

    def test_ask_questions_defaults_unanswered_without_hook(self):
        session = RecordingSession()
        self.assertEqual(session.ask_questions([{"question": "q"}]), "Unanswered")

    def test_ask_questions_uses_hook_when_set(self):
        session = RecordingSession()
        session.ask_fn = lambda questions: "answer"
        self.assertEqual(session.ask_questions([{"question": "q"}]), "answer")


class TestPlanModeGuard(unittest.TestCase):
    """Plan-mode tool guard: Bash always blocked, other writes blocked
    unless they target the plan file."""

    def make_plan_session(self):
        session = RecordingSession()
        session.switch_to_plan()
        return session

    def test_bash_blocked_in_plan_mode(self):
        session = self.make_plan_session()
        result = AgentSession.execute_tool(session, "Bash", {"command": "ls"})
        self.assertIn("blocked by plan mode", result)
        self.assertIn("Bash is disabled", result)

    def test_plan_blocked_other_file(self):
        session = self.make_plan_session()
        msg = session._plan_blocked("Edit", {"path": "/tmp/other.py"})
        self.assertIn("only the plan file", msg)

    def test_plan_blocked_plan_file_allowed(self):
        session = self.make_plan_session()
        plan_file = session.plan_mode.plan_file
        self.assertIsNone(session._plan_blocked("Edit", {"path": plan_file}))

    def test_plan_blocked_unknown_tool_no_path(self):
        session = self.make_plan_session()
        self.assertIsNone(session._plan_blocked("Read", {"path": "/x"}))

    def test_tool_path_resolution(self):
        session = RecordingSession()
        self.assertEqual(session._tool_path("Edit", {"path": "/a/b.py"}), "/a/b.py")
        self.assertEqual(session._tool_path("Insert", {"path": "/a/c.py"}), "/a/c.py")
        self.assertEqual(
            session._tool_path("Mkdir", {"parent": "/a", "name": "d"}),
            os.path.join("/a", "d"),
        )
        self.assertIsNone(session._tool_path("Read", {"path": "/x"}))


class TestFindSkill(unittest.TestCase):
    """find_skill resolves SKILL.md subdirectories, flat files, and
    None when the skill is absent or no skill dir is configured."""

    def test_no_skill_dir_returns_none(self):
        session = RecordingSession()
        session._skill_dir = None
        self.assertIsNone(session.find_skill("anything"))

    def test_subdirectory_skill_md(self):
        session = RecordingSession()
        with tempfile.TemporaryDirectory(prefix="pah-skills-") as d:
            sub = os.path.join(d, "rules")
            os.makedirs(sub)
            with open(os.path.join(sub, "SKILL.md"), "w") as f:
                f.write("# Rules")
            session._skill_dir = d
            self.assertEqual(
                session.find_skill("rules"), os.path.join(sub, "SKILL.md")
            )

    def test_flat_skill_file(self):
        session = RecordingSession()
        with tempfile.TemporaryDirectory(prefix="pah-skills-") as d:
            with open(os.path.join(d, "style.md"), "w") as f:
                f.write("# Style")
            with open(os.path.join(d, "extra.txt"), "w") as f:
                f.write("txt")
            session._skill_dir = d
            self.assertEqual(session.find_skill("style"), os.path.join(d, "style.md"))
            self.assertEqual(session.find_skill("extra"), os.path.join(d, "extra.txt"))

    def test_missing_skill_returns_none(self):
        session = RecordingSession()
        with tempfile.TemporaryDirectory(prefix="pah-skills-") as d:
            session._skill_dir = d
            self.assertIsNone(session.find_skill("nope"))


class TestAutoSaveDisabled(unittest.TestCase):
    def test_auto_save_skipped_when_disabled(self):
        session = RecordingSession()
        with mock.patch.object(config, "AUTO_SAVE_SESSION", False):
            with mock.patch.object(session.store, "save") as save:
                session.auto_save([Message(role="user", content="hi")], None)
        save.assert_not_called()


class TestTitleGeneration(unittest.TestCase):
    """Session-title generation: reasoning preamble stripped (even when
    it is not a prefix), failures logged and non-fatal."""

    def make_session(self, chat_sync):
        session = RecordingSession()
        session.logs = []
        session.log_fn = session.logs.append
        session.store.remember_first_user_message("Build a todo app please")
        client = FakeClient([])
        client.chat_sync = chat_sync
        session.client = client
        return session

    def test_reasoning_preamble_stripped_when_not_prefix(self):
        """Reasoning text appearing inside the title (not just as a
        prefix) is removed before sanitization."""

        def chat_sync(messages, system=None, temperature=None, max_tokens=None,
                      reasoning_effort=None):
            return Message(
                role="assistant", content="Title XYZ here", reasoning="XYZ"
            ), Usage()

        session = self.make_session(chat_sync)
        session.generate_session_title()
        self.assertEqual(session.store.title, "Title-here")
        self.assertFalse(session.store.title_pending)

    def test_title_failure_is_logged(self):
        def chat_sync(messages, system=None, temperature=None, max_tokens=None,
                      reasoning_effort=None):
            raise RuntimeError("api down")

        session = self.make_session(chat_sync)
        session.generate_session_title()
        self.assertIn("title generation failed: api down", session.logs[-1])
        self.assertFalse(session.store.title_pending)


class TestCancel(unittest.TestCase):
    def test_cancel_tolerates_abort_failure(self):
        session = RecordingSession()

        class AbortRaisingClient:
            def abort(self):
                raise RuntimeError("abort failed")

        session.client = AbortRaisingClient()
        session.cancel()  # must not raise
        self.assertTrue(session.cancel_event.is_set())
        self.assertEqual(session.cancel_generation, 1)


class TestCompactConversation(unittest.TestCase):
    """Manual /compact failure paths: empty history, concurrent
    compaction, empty summary, client exceptions."""

    def make_session(self, chat_sync):
        session = RecordingSession()
        session.logs = []
        session.log_fn = session.logs.append
        session.client = FakeClient([])
        session.client.chat_sync = chat_sync
        return session

    def test_nothing_to_compact(self):
        session = RecordingSession()
        ok, msg = session.compact_conversation()
        self.assertFalse(ok)
        self.assertEqual(msg, "Nothing to compact.")

    def test_compaction_in_progress(self):
        session = RecordingSession()
        session.last_messages = [Message(role="user", content="hi")]
        session.compacting = True
        ok, msg = session.compact_conversation()
        self.assertFalse(ok)
        self.assertEqual(msg, "Compaction already in progress.")

    def test_empty_summary_fails(self):
        def chat_sync(messages, system=None, temperature=None, max_tokens=None,
                      reasoning_effort=None):
            return Message(role="assistant", content=""), Usage()

        session = self.make_session(chat_sync)
        session.last_messages = [Message(role="user", content="hi")]
        ok, msg = session.compact_conversation()
        self.assertFalse(ok)
        self.assertEqual(msg, "Compaction failed: empty summary.")
        self.assertFalse(session.compacting)

    def test_client_failure_logged(self):
        def chat_sync(messages, system=None, temperature=None, max_tokens=None,
                      reasoning_effort=None):
            raise RuntimeError("boom")

        session = self.make_session(chat_sync)
        session.last_messages = [Message(role="user", content="hi")]
        ok, msg = session.compact_conversation()
        self.assertFalse(ok)
        self.assertIn("compaction failed: boom", session.logs[-1])
        self.assertEqual(msg, "Compaction failed: boom")
        self.assertFalse(session.compacting)


class TestSummarizeConversation(unittest.TestCase):
    """Manual /summary failure paths: empty history, client exceptions,
    empty responses."""

    def test_nothing_to_summarize(self):
        session = RecordingSession()
        self.assertEqual(session.summarize_conversation(), "Nothing to summarize.")

    def test_client_failure(self):
        def chat_sync(messages, system=None, temperature=None, max_tokens=None,
                      reasoning_effort=None):
            raise RuntimeError("boom")

        session = RecordingSession()
        session.client = FakeClient([])
        session.client.chat_sync = chat_sync
        session.last_messages = [Message(role="user", content="hi")]
        self.assertEqual(session.summarize_conversation(), "Summary failed: boom")

    def test_empty_response(self):
        def chat_sync(messages, system=None, temperature=None, max_tokens=None,
                      reasoning_effort=None):
            return Message(role="assistant", content=""), Usage()

        session = RecordingSession()
        session.client = FakeClient([])
        session.client.chat_sync = chat_sync
        session.last_messages = [Message(role="user", content="hi")]
        self.assertEqual(
            session.summarize_conversation(), "Summary failed: empty response."
        )


class TestTodos(unittest.TestCase):
    """update_todos / clear_todos mirror the current list into the
    session and notify the TUI panel."""

    def test_update_todos_stores_and_notifies(self):
        session = RecordingSession()
        notified = []
        session.notify_fn = notified.append
        session.update_todos([{"task": "a"}, {"task": "b"}])
        self.assertEqual(session.todos, [{"task": "a"}, {"task": "b"}])
        self.assertEqual(notified, ["todos"])

    def test_clear_todos_drops_and_notifies(self):
        session = RecordingSession()
        notified = []
        session.notify_fn = notified.append
        session.todos = [{"task": "a"}]
        session.clear_todos()
        self.assertEqual(session.todos, [])
        self.assertEqual(notified, ["todos"])


class TestPlanExit(unittest.TestCase):
    """PlanExit: approved switches to build and queues the approved
    message; rejected stays in plan; no-op outside plan mode."""

    def test_plan_exit_approved_switches_to_build(self):
        session = RecordingSession()
        session.switch_to_plan()
        session.confirm_fn = lambda prompt: True
        result = session.plan_exit()
        self.assertIn("approved switching to build", result)
        self.assertFalse(session.plan_mode.is_plan)
        self.assertGreater(len(session.pending_user_prompts), 0)
        names = [spec.name for spec in session.registry.specs()]
        self.assertNotIn("PlanExit", names)  # unregistered on the switch


class TestSwitchModes(unittest.TestCase):
    def test_switch_to_build_unregisters_plan_exit(self):
        session = RecordingSession()
        session.switch_to_plan()
        session.switch_to_build()
        self.assertFalse(session.plan_mode.is_plan)
        names = [spec.name for spec in session.registry.specs()]
        self.assertNotIn("PlanExit", names)

    def test_switch_to_plan_registers_plan_exit(self):
        session = RecordingSession()
        session.switch_to_plan()
        self.assertTrue(session.plan_mode.is_plan)
        self.assertIsNotNone(session.plan_mode.plan_file)
        names = [spec.name for spec in session.registry.specs()]
        self.assertIn("PlanExit", names)


class TestClose(unittest.TestCase):
    """close() cancels the run, marks the session dead, cleans up the
    plan file and closes the client."""

    def test_close_cancels_and_cleans_up(self):
        session = RecordingSession()
        session.switch_to_plan()  # creates a real plan file
        plan_file = session.plan_mode.plan_file
        self.assertTrue(os.path.exists(plan_file))

        class ClosableClient:
            closed = False

            def close(self):
                ClosableClient.closed = True

        session.client = ClosableClient()
        session.close()
        self.assertFalse(session.alive)
        self.assertTrue(session.cancel_event.is_set())
        self.assertFalse(os.path.exists(plan_file))
        self.assertTrue(ClosableClient.closed)


if __name__ == "__main__":
    unittest.main()
