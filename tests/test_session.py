"""Session unit tests: dir discovery, plan-mode guards, skills,
auto-save/title/cancel edge cases, compact/summarize failure paths."""

import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))

import plan_cleanup  # noqa: F401,E402  (side-effect: auto-remove /tmp plan dirs)
from agent.agent_test_utils import FakeClient, RecordingSession

from python_agent_harness import config
from python_agent_harness.models import Message, Usage
from python_agent_harness.session import (
    Session,
    find_context_dir,
    find_skill_dir,
)


class TestFindDirs(unittest.TestCase):
    """find_skill_dir / find_context_dir resolution: configured path
    wins, then the project's own directory, then None.  Nothing outside
    the project is discovered implicitly — it must be configured."""

    def test_skill_dir_configured_wins(self):
        with tempfile.TemporaryDirectory(prefix="pah-skills-") as d:
            self.assertEqual(find_skill_dir("/proj", d), d)

    def test_skill_dir_default_project_fallback(self):
        """A configured-but-missing path falls through to the project's
        skills/ directory."""
        with tempfile.TemporaryDirectory(prefix="pah-proj-") as proj:
            skills = os.path.join(proj, "skills")
            os.makedirs(skills)
            self.assertEqual(find_skill_dir(proj, "/nonexistent"), skills)

    def test_skill_dir_ignores_home_directory(self):
        """``~/.emacs.d/skills`` is NOT discovered implicitly any more."""
        with tempfile.TemporaryDirectory(prefix="pah-home-") as home:
            os.makedirs(os.path.join(home, ".emacs.d", "skills"))
            with mock.patch.dict(os.environ, {"HOME": home}):
                self.assertIsNone(find_skill_dir("/proj", None))

    def test_skill_dir_none_when_missing(self):
        with tempfile.TemporaryDirectory(prefix="pah-proj-") as proj:
            self.assertIsNone(find_skill_dir(proj, None))

    def test_context_dir_configured_wins(self):
        with tempfile.TemporaryDirectory(prefix="pah-ctx-") as d:
            self.assertEqual(find_context_dir("/proj", d), d)

    def test_context_dir_project_fallback(self):
        with tempfile.TemporaryDirectory(prefix="pah-proj-") as proj:
            ctx = os.path.join(proj, "contexts")
            os.makedirs(ctx)
            self.assertEqual(find_context_dir(proj, None), ctx)

    def test_context_dir_ignores_home_directory(self):
        """``~/.emacs.d/contexts`` is NOT discovered implicitly any more."""
        with tempfile.TemporaryDirectory(prefix="pah-home-") as home:
            os.makedirs(os.path.join(home, ".emacs.d", "contexts"))
            with mock.patch.dict(os.environ, {"HOME": home}):
                self.assertIsNone(find_context_dir("/proj", None))

    def test_context_dir_none_when_missing(self):
        with tempfile.TemporaryDirectory(prefix="pah-proj-") as proj:
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
        result = Session.execute_tool(session, "Bash", {"command": "ls"})
        self.assertIn("blocked by plan mode", result)
        self.assertIn("Bash is disabled", result)

    def test_plan_blocked_other_file(self):
        session = self.make_plan_session()
        msg = session._plan_blocked("Edit", {"path": "/tmp/other.py"})
        self.assertIn("blocked by plan mode", msg)

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
    """find_skill resolves SKILL.md files by their frontmatter name
    (opencode-style): the directory name is irrelevant, and flat files
    without frontmatter are not skills."""

    def test_no_skill_dir_returns_none(self):
        session = RecordingSession()
        session._skill_dir = None
        self.assertIsNone(session.find_skill("anything"))

    def test_skill_found_by_frontmatter_name(self):
        session = RecordingSession()
        with tempfile.TemporaryDirectory(prefix="pah-skills-") as d:
            sub = os.path.join(d, "rules")
            os.makedirs(sub)
            with open(os.path.join(sub, "SKILL.md"), "w") as f:
                f.write("---\nname: my-rules\ndescription: rules\n---\n# Rules")
            session._skill_dir = d
            self.assertEqual(session.find_skill("my-rules"), os.path.join(sub, "SKILL.md"))

    def test_frontmatter_name_differs_from_directory_name(self):
        """Regression test: the advertised name (frontmatter) must
        resolve even when the directory name differs (e.g.
        skills/weather-forecaster/SKILL.md with name 天气预报助手)."""
        session = RecordingSession()
        with tempfile.TemporaryDirectory(prefix="pah-skills-") as d:
            sub = os.path.join(d, "weather-forecaster")
            os.makedirs(sub)
            with open(os.path.join(sub, "SKILL.md"), "w") as f:
                f.write("---\nname: 天气预报助手\n---\n# body")
            session._skill_dir = d
            self.assertEqual(session.find_skill("天气预报助手"), os.path.join(sub, "SKILL.md"))

    def test_nested_skill_dir_discovered(self):
        session = RecordingSession()
        with tempfile.TemporaryDirectory(prefix="pah-skills-") as d:
            nested = os.path.join(d, "a", "b", "deep")
            os.makedirs(nested)
            with open(os.path.join(nested, "SKILL.md"), "w") as f:
                f.write("---\nname: deep-skill\n---\nbody")
            session._skill_dir = d
            self.assertEqual(session.find_skill("deep-skill"), os.path.join(nested, "SKILL.md"))

    def test_flat_file_without_frontmatter_returns_none(self):
        """Flat files (skills/style.md) are no longer resolvable: only
        SKILL.md files with a frontmatter name are skills."""
        session = RecordingSession()
        with tempfile.TemporaryDirectory(prefix="pah-skills-") as d:
            with open(os.path.join(d, "style.md"), "w") as f:
                f.write("# Style")
            with open(os.path.join(d, "extra.txt"), "w") as f:
                f.write("txt")
            session._skill_dir = d
            self.assertIsNone(session.find_skill("style"))
            self.assertIsNone(session.find_skill("extra"))

    def test_missing_skill_returns_none(self):
        session = RecordingSession()
        with tempfile.TemporaryDirectory(prefix="pah-skills-") as d:
            session._skill_dir = d
            self.assertIsNone(session.find_skill("nope"))

    def test_symlinked_skill_dir_resolves(self):
        """A skill dir that is a symlink pointing outside the skill root
        must still resolve (the index scans with symlinks enabled)."""
        session = RecordingSession()
        with (
            tempfile.TemporaryDirectory(prefix="pah-skills-") as d,
            tempfile.TemporaryDirectory(prefix="pah-skills-out-") as outside,
        ):
            sub = os.path.join(outside, "linked")
            os.makedirs(sub)
            with open(os.path.join(sub, "SKILL.md"), "w") as f:
                f.write("---\nname: linked-skill\n---\n# Linked")
            os.symlink(sub, os.path.join(d, "linked"))
            session._skill_dir = d
            self.assertEqual(session.find_skill("linked-skill"), os.path.join(sub, "SKILL.md"))

    def test_traversal_skill_returns_none(self):
        """Path-traversal inputs are inert: lookup only matches names in
        the scanned index."""
        session = RecordingSession()
        with tempfile.TemporaryDirectory(prefix="pah-skills-") as d:
            session._skill_dir = d
            self.assertIsNone(session.find_skill("../etc"))
            self.assertIsNone(session.find_skill("a/../../etc"))


class TestAutoSaveDisabled(unittest.TestCase):
    def test_auto_save_skipped_when_disabled(self):
        session = RecordingSession()
        with (
            mock.patch.object(config, "AUTO_SAVE_SESSION", False),
            mock.patch.object(session.store, "save") as save,
        ):
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

        def chat_sync(
            messages, system=None, temperature=None, max_tokens=None, reasoning_effort=None
        ):
            return Message(role="assistant", content="Title XYZ here", reasoning="XYZ"), Usage()

        session = self.make_session(chat_sync)
        session.generate_session_title()
        self.assertEqual(session.store.title, "Title-here")
        self.assertFalse(session.store.title_pending)

    def test_title_failure_is_logged(self):
        def chat_sync(
            messages, system=None, temperature=None, max_tokens=None, reasoning_effort=None
        ):
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

    def test_cancel_aborts_subagent_client_too(self):
        """A dedicated sub-agent client (separate subagent_llm) streams
        on its own pool — cancel must abort it as well, so a blocked
        sub-agent read is interrupted (and one shared client is not
        aborted twice)."""
        session = RecordingSession()

        class SpyClient:
            def __init__(self):
                self.aborted = 0

            def abort(self):
                self.aborted += 1

        main = SpyClient()
        sub = SpyClient()
        session.client = main
        session.subagent_client = sub
        session.cancel()
        self.assertEqual(main.aborted, 1)
        self.assertEqual(sub.aborted, 1)

        # when the sub-agent shares the main client, abort runs once
        session2 = RecordingSession()
        session2.client = main
        session2.subagent_client = main
        session2.cancel()
        self.assertEqual(main.aborted, 2)


class TestSubagentDedicatedClient(unittest.TestCase):
    """Each Agent invocation runs on a dedicated client clone: the
    session's sub-agent client is a TEMPLATE, never shared by
    concurrent sub-agents.  A shared Client would race — _reset_http/
    abort swap and close the one httpx pool, and _aborted is per-request
    state — so one sub-agent's connection failure (or a Ctrl-C) could
    tear down a sibling's in-flight request."""

    def _run_concurrently(self, session, n=2):
        import threading
        from unittest import mock

        entered = threading.Barrier(n)
        seen = []
        pools_open = []

        def fake_run_subagent(session_, description, prompt, client=None):
            seen.append(client)
            pools_open.append(not client._http.is_closed)
            entered.wait(timeout=5)
            return f"done-{description}"

        with mock.patch("python_agent_harness.session.run_subagent", side_effect=fake_run_subagent):
            threads = [
                threading.Thread(target=lambda i=i: session.run_subagent("subagent", f"t{i}", "p"))
                for i in range(n)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)
            self.assertEqual([t.is_alive() for t in threads], [False, False])
        return seen, pools_open

    def test_concurrent_invocations_get_distinct_clients(self):
        """Concurrent sub-agents never share the session client: each
        invocation gets its own clone, distinct from the others and
        from the shared (default) session client."""
        from python_agent_harness.client import Client

        main = Client(base_url="http://127.0.0.1:1/v1", api_key="k", model="m", timeout=1)
        session = RecordingSession()
        session.client = main
        session.subagent_client = main  # no separate subagent LLM: the old shared case
        try:
            seen, pools_open = self._run_concurrently(session)
            self.assertEqual(len(seen), 2)
            for c in seen:
                self.assertIsNot(c, main)
            self.assertIsNot(seen[0], seen[1])
            # every clone is its own open pool while in flight (a
            # shared pool was the race)
            self.assertEqual(pools_open, [True, True])
        finally:
            main.close()

    def test_active_clients_tracked_aborted_on_cancel_and_released(self):
        """While a sub-agent is in flight its dedicated client is
        tracked; session.cancel() aborts every active clone (a blocked
        sub-agent read is interrupted on Ctrl-C); completion releases
        the clone and closes its pool."""
        import threading
        from unittest import mock

        from python_agent_harness.client import Client

        main = Client(base_url="http://127.0.0.1:1/v1", api_key="k", model="m", timeout=1)
        session = RecordingSession()
        session.client = main
        session.subagent_client = main

        entered = threading.Event()
        release = threading.Event()
        seen = []

        def fake_run_subagent(session_, description, prompt, client=None):
            seen.append(client)
            entered.set()
            release.wait(timeout=10)
            return "done"

        try:
            with mock.patch(
                "python_agent_harness.session.run_subagent", side_effect=fake_run_subagent
            ):
                t = threading.Thread(target=lambda: session.run_subagent("subagent", "t0", "p"))
                t.start()
                self.assertTrue(entered.wait(timeout=5))
                clone = seen[0]
                self.assertIn(clone, session._active_subagent_clients)
                session.cancel()
                # abort() swapped the clone's pool and set its flag
                self.assertTrue(clone._aborted)
                release.set()
                t.join(timeout=5)
                self.assertFalse(t.is_alive())
                # released on completion: no stragglers, pool closed
                self.assertNotIn(clone, session._active_subagent_clients)
                self.assertTrue(clone._http.is_closed)
        finally:
            release.set()
            main.close()

    def test_session_close_releases_inflight_clients(self):
        """close() (defensive path) closes clones still tracked from
        workers that have not finished winding down."""
        import threading
        from unittest import mock

        from python_agent_harness.client import Client

        main = Client(base_url="http://127.0.0.1:1/v1", api_key="k", model="m", timeout=1)
        session = RecordingSession()
        session.client = main
        session.subagent_client = main

        entered = threading.Event()
        release = threading.Event()
        seen = []

        def fake_run_subagent(session_, description, prompt, client=None):
            seen.append(client)
            entered.set()
            release.wait(timeout=10)
            return "done"

        try:
            with mock.patch(
                "python_agent_harness.session.run_subagent", side_effect=fake_run_subagent
            ):
                t = threading.Thread(target=lambda: session.run_subagent("subagent", "t0", "p"))
                t.start()
                self.assertTrue(entered.wait(timeout=5))
                clone = seen[0]
                session.close()
                self.assertTrue(clone._http.is_closed)
                self.assertEqual(session._active_subagent_clients, [])
                release.set()
                t.join(timeout=5)
                self.assertFalse(t.is_alive())
        finally:
            release.set()
            main.close()

    def test_non_client_subagent_client_passed_through(self):
        """A non-Client subagent_client (test doubles / custom clients)
        is used as-is: no clone, no tracking, no closing."""
        from unittest import mock

        session = RecordingSession()
        stub = object()
        session.subagent_client = stub
        seen = []

        def fake_run_subagent(session_, description, prompt, client=None):
            seen.append(client)
            return "done"

        with mock.patch("python_agent_harness.session.run_subagent", side_effect=fake_run_subagent):
            result = session.run_subagent("subagent", "t0", "p")
        self.assertEqual(result, "done")
        self.assertIs(seen[0], stub)
        self.assertEqual(session._active_subagent_clients, [])

    def test_clone_inherits_subagent_own_llm_config(self):
        """The per-invocation clone is cloned from the SUB-AGENT client,
        not the main one: when a separate subagent_llm is configured
        (different model/base_url/api_key/timeout), every sub-agent runs
        on the sub-agent's OWN LLM settings — never the main model."""
        from python_agent_harness.client import Client

        main = Client(
            base_url="https://main.example/v1",
            api_key="sk-main",
            model="main-model",
            timeout=600,
        )
        sub = Client(
            base_url="https://sub.example/v1",
            api_key="sk-sub",
            model="cheap-model",
            timeout=300,
        )
        session = RecordingSession()
        session.client = main
        session.subagent_client = sub
        try:
            clone, owned = session._new_subagent_client()
            self.assertTrue(owned)
            self.assertIsNot(clone, sub)
            self.assertIsNot(clone, main)
            # the clone carries the SUB-AGENT's LLM, not the main one
            self.assertEqual(clone.model, "cheap-model")
            self.assertEqual(clone.base_url, "https://sub.example/v1")
            self.assertEqual(clone.api_key, "sk-sub")
            self.assertEqual(clone.timeout, 300)
            clone.close()
        finally:
            main.close()
            sub.close()

    def test_clone_inherits_main_config_when_no_subagent_llm(self):
        """Without a separate subagent_llm the template IS the main
        client, so the clone inherits the main settings — the sub-agent
        path is then identical to the main agent's."""
        from python_agent_harness.client import Client

        main = Client(
            base_url="https://main.example/v1",
            api_key="sk-main",
            model="main-model",
            timeout=600,
        )
        session = RecordingSession()
        session.client = main
        session.subagent_client = main  # unset subagent_llm: shares the main client
        try:
            clone, owned = session._new_subagent_client()
            self.assertTrue(owned)
            self.assertEqual(clone.model, "main-model")
            self.assertEqual(clone.base_url, "https://main.example/v1")
            self.assertEqual(clone.api_key, "sk-main")
            self.assertEqual(clone.timeout, 600)
            clone.close()
        finally:
            main.close()


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
        def chat_sync(
            messages, system=None, temperature=None, max_tokens=None, reasoning_effort=None
        ):
            return Message(role="assistant", content=""), Usage()

        session = self.make_session(chat_sync)
        session.last_messages = [Message(role="user", content="hi")]
        ok, msg = session.compact_conversation()
        self.assertFalse(ok)
        self.assertEqual(msg, "Compaction failed: empty summary.")
        self.assertFalse(session.compacting)

    def test_client_failure_logged(self):
        def chat_sync(
            messages, system=None, temperature=None, max_tokens=None, reasoning_effort=None
        ):
            raise RuntimeError("boom")

        session = self.make_session(chat_sync)
        session.last_messages = [Message(role="user", content="hi")]
        ok, msg = session.compact_conversation()
        self.assertFalse(ok)
        self.assertIn("compaction failed: boom", session.logs[-1])
        self.assertEqual(msg, "Compaction failed: boom")
        self.assertFalse(session.compacting)

    def test_success_keeps_every_user_prompt(self):
        """A successful manual /compact replaces the history with the
        summary frame followed by every prompt; nudges are excluded but
        harness-injected plan/build reminders are kept (mode context
        must survive compaction)."""
        session = RecordingSession()
        session.tools_enabled = False
        session.last_messages = [
            Message(role="user", content="first"),
            Message(role="assistant", content="ok"),
            Message(role="user", content="second"),
            Message(role="user", content=config.NUDGE_MESSAGE, injected=True),
            Message(
                role="user",
                content="The plan at /tmp/x/PLAN.md has been approved, "
                "you can now edit files. Execute the plan",
                injected=True,
            ),
        ]
        ok, msg = session.compact_conversation()
        self.assertTrue(ok)
        self.assertEqual(msg, "Buffer compacted successfully.")
        self.assertEqual([m.role for m in session.last_messages], ["user", "user", "user", "user"])
        self.assertTrue(session.last_messages[0].text().startswith("**[Compacted Summary]**"))
        self.assertEqual(
            [m.text() for m in session.last_messages[1:]],
            [
                "first",
                "second",
                "The plan at /tmp/x/PLAN.md has been approved, "
                "you can now edit files. Execute the plan",
            ],
        )


class TestSummarizeConversation(unittest.TestCase):
    """Manual /summary failure paths: empty history, client exceptions,
    empty responses."""

    def test_nothing_to_summarize(self):
        session = RecordingSession()
        self.assertEqual(session.summarize_conversation(), "Nothing to summarize.")

    def test_client_failure(self):
        def chat_sync(
            messages, system=None, temperature=None, max_tokens=None, reasoning_effort=None
        ):
            raise RuntimeError("boom")

        session = RecordingSession()
        session.client = FakeClient([])
        session.client.chat_sync = chat_sync
        session.last_messages = [Message(role="user", content="hi")]
        self.assertEqual(session.summarize_conversation(), "Summary failed: boom")

    def test_empty_response(self):
        def chat_sync(
            messages, system=None, temperature=None, max_tokens=None, reasoning_effort=None
        ):
            return Message(role="assistant", content=""), Usage()

        session = RecordingSession()
        session.client = FakeClient([])
        session.client.chat_sync = chat_sync
        session.last_messages = [Message(role="user", content="hi")]
        self.assertEqual(session.summarize_conversation(), "Summary failed: empty response.")


class TestTodos(unittest.TestCase):
    """update_todos / clear_todos mirror the current list into the
    session and notify the TUI panel."""

    def test_update_todos_stores_and_notifies(self):
        session = RecordingSession()
        notified = []
        session.notify_fn = lambda kind, data=None: notified.append(kind)
        session.update_todos([{"task": "a"}, {"task": "b"}])
        self.assertEqual(session.todos, [{"task": "a"}, {"task": "b"}])
        self.assertEqual(notified, ["todos"])

    def test_clear_todos_drops_and_notifies(self):
        session = RecordingSession()
        notified = []
        session.notify_fn = lambda kind, data=None: notified.append(kind)
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
    """close() cancels the run, marks the session dead and closes the
    client; the plan file is kept (one per session) for later
    reference."""

    def test_close_cancels_and_keeps_plan_file(self):
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
        self.assertTrue(os.path.exists(plan_file))
        self.assertTrue(ClosableClient.closed)

    def test_close_closes_subagent_client_too(self):
        """A dedicated sub-agent client must be closed with the session."""
        session = RecordingSession()

        class ClosableClient:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        main = ClosableClient()
        sub = ClosableClient()
        session.client = main
        session.subagent_client = sub
        session.close()
        self.assertTrue(main.closed)
        self.assertTrue(sub.closed)


class TestModelSwitching(unittest.TestCase):
    """Test /model command functionality for switching between LLM profiles."""

    def test_switch_model_success(self):
        """Switching to a configured model profile updates client and session."""
        session = RecordingSession(
            model_profiles={
                "deepseek": {
                    "base_url": "https://api.deepseek.com/v1",
                    "api_key": "sk-deepseek",
                    "model": "deepseek-v4-flash",
                    "temperature": 0.0,
                },
                "glm": {
                    "base_url": "https://api-inference.modelscope.cn/v1",
                    "api_key": "ms-key",
                    "model": "ZhipuAI/GLM-5.2",
                    "temperature": 0.0,
                },
            }
        )
        success, msg = session.switch_model("deepseek")
        self.assertTrue(success)
        self.assertIn("switched to deepseek", msg)
        self.assertEqual(session.model, "deepseek-v4-flash")
        self.assertEqual(session.client.model, "deepseek-v4-flash")
        self.assertEqual(session.client.base_url, "https://api.deepseek.com/v1")
        self.assertEqual(session.client.api_key, "sk-deepseek")

    def test_switch_model_unknown_name_fails(self):
        """Switching to an unknown model name returns error message."""
        session = RecordingSession(
            model_profiles={
                "deepseek": {"model": "deepseek-chat"},
                "glm": {"model": "glm-model"},
            }
        )
        success, msg = session.switch_model("unknown")
        self.assertFalse(success)
        self.assertIn("unknown model", msg)
        self.assertIn("deepseek", msg)
        self.assertIn("glm", msg)

    def test_switch_model_re_resolves_context_window(self):
        """Switching models must make the next context-window access
        resolve for the NEW model: the ratio computation divides by
        the new model's window, not the old one."""
        from python_agent_harness.client import Client

        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "config.json")
            with open(p, "w", encoding="utf-8") as f:
                f.write('{"context_windows": {"deepseek-v4*": 1000000}}')
            client = Client(base_url="http://x/v1", api_key="k", model="gpt-5-mini", config_path=p)
            self.addCleanup(client.close)
            session = RecordingSession(model_profiles={"deepseek": {"model": "deepseek-v4-flash"}})
            session.client = client
            self.assertEqual(client.context_window, 128_000)
            success, _ = session.switch_model("deepseek")
            self.assertTrue(success)
            self.assertEqual(client.context_window, 1_000_000)

    def test_switch_model_resets_calibrator(self):
        """Switching models must drop the token-calibration factor: it
        was tuned to the previous model's tokenizer and would skew the
        first context estimates for the new model."""
        session = RecordingSession(model_profiles={"deepseek": {"model": "deepseek-v4-flash"}})
        session.calibrator.factor = 2.5
        session.calibrator.last_raw_estimate = 1234
        success, _ = session.switch_model("deepseek")
        self.assertTrue(success)
        self.assertEqual(session.calibrator.factor, 1.0)
        self.assertIsNone(session.calibrator.last_raw_estimate)

    def test_switch_model_preserves_conversation_history(self):
        """Switching models does not clear conversation history."""
        from python_agent_harness.models import Message

        session = RecordingSession(
            model_profiles={
                "deepseek": {"model": "deepseek-chat"},
                "glm": {"model": "glm-model"},
            }
        )
        # Add some conversation history
        session.last_messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ]

        # Switch model
        success, _ = session.switch_model("deepseek")
        self.assertTrue(success)

        # Conversation history should be preserved
        self.assertEqual(len(session.last_messages), 2)
        self.assertEqual(session.last_messages[0].role, "user")
        self.assertEqual(session.last_messages[1].role, "assistant")

    def test_switch_model_updates_temperature(self):
        """Switching model updates temperature setting."""
        session = RecordingSession(
            model_profiles={
                "high_temp": {"model": "test", "temperature": 0.8},
            }
        )
        self.assertEqual(session.temperature, 0.0)  # default

        success, _ = session.switch_model("high_temp")
        self.assertTrue(success)
        self.assertEqual(session.temperature, 0.8)

    def test_no_model_profiles_returns_error(self):
        """When no model profiles are configured, switch_model fails
        (``default`` is still advertised)."""
        session = RecordingSession(model_profiles={})
        success, msg = session.switch_model("any")
        self.assertFalse(success)
        self.assertIn("default", msg)

    def test_switch_model_default_restores_original_settings(self):
        """``switch_model("default")`` restores the session-start main
        llm settings after switching to profiles, so the original model
        stays reachable."""
        session = RecordingSession(
            model_profiles={
                "deepseek": {
                    "base_url": "https://api.deepseek.com/v1",
                    "api_key": "sk-deepseek",
                    "model": "deepseek-v4-flash",
                    "temperature": 0.0,
                },
            },
            llm_settings={
                "base_url": "https://llm.example.com/v1",
                "api_key": "llm-key",
                "model": "base-model",
                "backend": "OpenAI-compatible",
                "temperature": 0.2,
                "max_tokens": None,
                "timeout": 600.0,
                "reasoning_effort": None,
                "stream": True,
            },
        )
        self.assertEqual(session.model, "gpt-5-mini")  # RecordingSession's hardcoded model
        success, msg = session.switch_model("deepseek")
        self.assertTrue(success)
        self.assertEqual(session.model, "deepseek-v4-flash")
        self.assertEqual(session.temperature, 0.0)
        success, msg = session.switch_model("default")
        self.assertTrue(success)
        self.assertIn("switched to default", msg)
        # default restores the main llm settings (base-model), not the
        # hardcoded construction default
        self.assertEqual(session.model, "base-model")
        self.assertEqual(session.client.model, "base-model")
        self.assertEqual(session.client.base_url, "https://llm.example.com/v1")
        self.assertEqual(session.client.api_key, "llm-key")
        self.assertEqual(session.temperature, 0.2)

    def test_profile_settings_override_llm_settings(self):
        """Profile settings take precedence over the main llm settings."""
        session = RecordingSession(
            model_profiles={
                "fast": {
                    "model": "fast-model",
                    "temperature": 0.5,
                    "base_url": "https://fast.example/v1",
                },
            },
            llm_settings={
                "base_url": "https://llm.example.com/v1",
                "api_key": "llm-key",
                "model": "base-model",
                "backend": "OpenAI-compatible",
                "temperature": 0.0,
                "max_tokens": None,
                "timeout": 600.0,
                "reasoning_effort": None,
                "stream": True,
            },
        )
        success, _ = session.switch_model("fast")
        self.assertTrue(success)
        self.assertEqual(session.model, "fast-model")
        self.assertEqual(session.client.model, "fast-model")
        self.assertEqual(session.temperature, 0.5)
        self.assertEqual(session.client.base_url, "https://fast.example/v1")
        # Unset keys inherit the main llm settings, not the defaults
        self.assertEqual(session.client.api_key, "llm-key")

    def test_switch_model_unset_keys_fall_back_to_llm_settings(self):
        """After switching, unset profile keys revert to llm settings (no drift)."""
        session = RecordingSession(
            model_profiles={
                "deepseek": {"base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat"},
                "glm": {"model": "glm-5.2"},
            },
            llm_settings={
                "base_url": "https://llm.example.com/v1",
                "api_key": "llm-key",
                "model": "base-model",
                "backend": "OpenAI-compatible",
                "temperature": 0.0,
                "max_tokens": None,
                "timeout": 600.0,
                "reasoning_effort": None,
                "stream": True,
            },
        )
        session.switch_model("deepseek")
        self.assertEqual(session.client.base_url, "https://api.deepseek.com/v1")
        # glm sets no base_url -> falls back to llm settings, not deepseek's
        success, _ = session.switch_model("glm")
        self.assertTrue(success)
        self.assertEqual(session.client.base_url, "https://llm.example.com/v1")
        self.assertEqual(session.model, "glm-5.2")
        self.assertEqual(session.client.api_key, "llm-key")

    def test_switch_model_none_values_do_not_override(self):
        """Profile keys explicitly set to None inherit llm settings."""
        session = RecordingSession(
            model_profiles={
                "partial": {
                    "model": "partial-model",
                    "temperature": None,
                    "stream": None,
                    "reasoning_effort": None,
                },
            },
            llm_settings={
                "base_url": "https://llm.example.com/v1",
                "api_key": "llm-key",
                "model": "base-model",
                "backend": "OpenAI-compatible",
                "temperature": 0.7,
                "max_tokens": 4096,
                "timeout": 600.0,
                "reasoning_effort": "high",
                "stream": False,
            },
        )
        success, _ = session.switch_model("partial")
        self.assertTrue(success)
        self.assertEqual(session.model, "partial-model")
        # None values in the profile must inherit from llm settings
        self.assertEqual(session.temperature, 0.7)
        self.assertEqual(session.stream, False)
        self.assertEqual(session.reasoning_effort, "high")
        self.assertEqual(session.max_tokens, 4096)
        self.assertEqual(session.client.api_key, "llm-key")

    def test_switch_model_round_trip_restores_llm_settings(self):
        """Switching profile A -> profile B restores llm settings for keys B omits."""
        session = RecordingSession(
            model_profiles={
                "a": {"model": "a-model", "temperature": 0.9, "stream": False},
                "b": {"model": "b-model"},
            },
            llm_settings={
                "base_url": "https://llm.example.com/v1",
                "api_key": "llm-key",
                "model": "base-model",
                "backend": "OpenAI-compatible",
                "temperature": 0.7,
                "max_tokens": None,
                "timeout": 600.0,
                "reasoning_effort": None,
                "stream": True,
            },
        )
        session.switch_model("a")
        self.assertEqual(session.temperature, 0.9)
        self.assertEqual(session.stream, False)
        # b omits temperature/stream -> reverts to llm settings, not a's
        session.switch_model("b")
        self.assertEqual(session.model, "b-model")
        self.assertEqual(session.temperature, 0.7)
        self.assertEqual(session.stream, True)
        # switch back to a: a's values apply again
        session.switch_model("a")
        self.assertEqual(session.temperature, 0.9)
        self.assertEqual(session.stream, False)

    def test_switch_model_updates_timeout(self):
        """Switching to a profile with a different timeout applies it to
        the client's HTTP pool, not just the attribute."""
        session = RecordingSession(
            model_profiles={
                "slow": {"model": "slow-model", "timeout": 120.0},
            },
            llm_settings={
                "base_url": "https://llm.example.com/v1",
                "api_key": "llm-key",
                "model": "base-model",
                "backend": "OpenAI-compatible",
                "temperature": 0.0,
                "max_tokens": None,
                "timeout": 600.0,
                "reasoning_effort": None,
                "stream": True,
            },
        )
        self.assertEqual(session.client.timeout, 600.0)
        success, _ = session.switch_model("slow")
        self.assertTrue(success)
        self.assertEqual(session.client.timeout, 120.0)


if __name__ == "__main__":
    unittest.main()
