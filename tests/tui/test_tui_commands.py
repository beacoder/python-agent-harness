"""TUI slash-command tests (/init /review /explain, /sessions, /restore,
/model) — dispatch, argument parsing, session listing and model switching."""

import os
import sys
import tempfile
import unittest
import unittest.mock as mock

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import plan_cleanup  # noqa: F401,E402  (side-effect: auto-remove /tmp plan dirs)
from tui_test_utils import make_tui

from python_agent_harness.models import Message
from python_agent_harness.tui import Tui


class TestTuiCommands(unittest.TestCase):
    # ------------------------------------------------------------------
    # slash commands (/init /review /explain)
    # ------------------------------------------------------------------
    def test_slash_command_args_parsing(self):
        """Arg parsing matches the CLI signatures: [project] first, then
        the command's argument; a lone non-directory token is the
        argument (so `/review main` reviews the branch, not a project)."""
        tui, _ = make_tui()
        self.assertEqual(tui._command_args("init", ""), (None, None))
        self.assertEqual(tui._command_args("init", "myproj"), ("myproj", None))
        self.assertEqual(
            tui._command_args("init", 'myproj --extra "focus CI"'),
            ("myproj", "focus CI"),
        )
        self.assertEqual(tui._command_args("init", "--extra x"), (None, "x"))
        self.assertEqual(tui._command_args("review", ""), (None, None))
        self.assertEqual(tui._command_args("review", "main"), (None, "main"))
        self.assertEqual(tui._command_args("review", "abc123"), (None, "abc123"))
        self.assertEqual(tui._command_args("explain", "client.py"), (None, "client.py"))
        self.assertEqual(
            tui._command_args("explain", "the retry logic"),
            (None, "the retry logic"),
        )
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(tui._command_args("review", d), (d, None))
            self.assertEqual(tui._command_args("explain", d), (d, None))
            self.assertEqual(tui._command_args("review", f"{d} main"), (d, "main"))

    def test_slash_dispatch_runs_command_in_session(self):
        """/init, /review and /explain run their SessionCommand in the
        current session: the command prompt becomes the run's system
        prompt and the kickoff message is the user text."""
        tui, _ = make_tui()
        captured = {}

        def fake_start(text, system=None, restore=None):
            captured["text"] = text
            captured["system"] = system
            captured["restore"] = restore

        with mock.patch.object(tui, "_start_agent", side_effect=fake_start):
            self.assertFalse(tui._handle_slash("/init"))
            self.assertIn("AGENTS.md", captured["text"])
            self.assertIn("Create or update", captured["system"])

            tui._handle_slash("/review main")
            self.assertIn("Review the requested code changes", captured["text"])
            self.assertIn("code reviewer", captured["system"])
            self.assertIn("main", captured["system"])  # $ARGUMENTS substituted

            tui._handle_slash("/explain client.py")
            self.assertIn("instructions", captured["text"])  # custom kickoff
            self.assertIn("client.py", captured["system"])
            self.assertIn("explain", captured["system"])

    def test_slash_command_kickoff_anchored_when_history_exists(self):
        """Mid-conversation the generic kickoff would read as a
        continuation of the previous (already finished) task, so the
        command names itself and marks the earlier messages as
        background context.  An empty conversation keeps the original
        kickoff (fresh-session semantics)."""
        tui, _ = make_tui()  # make_tui starts with a non-empty conversation
        captured = {}

        def fake_start(text, system=None, restore=None):
            captured["text"] = text

        with mock.patch.object(tui, "_start_agent", side_effect=fake_start):
            tui._handle_slash("/explain client.py")
        self.assertIn("NEW /explain request: client.py", captured["text"])
        self.assertIn("background context", captured["text"])
        self.assertIn("Proceed with the task described in your instructions.", captured["text"])

        # empty conversation: kickoff stays the plain generic message
        tui.conversation_history = []
        tui.session.last_messages = []
        with mock.patch.object(tui, "_start_agent", side_effect=fake_start):
            tui._handle_slash("/explain client.py")
        self.assertEqual(
            captured["text"].strip(),
            "Proceed with the task described in your instructions.",
        )

    def test_slash_command_kickoff_anchored_without_target(self):
        """The anchor names the command even when the command has no
        arguments (the target is described in the prompt instead)."""
        tui, _ = make_tui()
        tui.conversation_history = [Message(role="user", content="old task")]
        captured = {}

        def fake_start(text, system=None, restore=None):
            captured["text"] = text

        with mock.patch.object(tui, "_start_agent", side_effect=fake_start):
            tui._handle_slash("/review")
        self.assertIn("NEW /review request", captured["text"])
        self.assertNotIn("NEW /review request:", captured["text"])

    def test_slash_command_project_borrowed_and_restored(self):
        """A project given to a slash command borrows the session's
        project dir for the run (tool cwd) and restores it afterwards."""
        tui, _ = make_tui()
        with tempfile.TemporaryDirectory() as d:

            def fake_start(text, system=None, restore=None):
                self.assertEqual(tui.session.project_dir, os.path.abspath(d))
                restore()  # simulate the run finishing

            with mock.patch.object(tui, "_start_agent", side_effect=fake_start):
                tui._handle_slash(f"/init {d}")
        self.assertEqual(tui.session.project_dir, "/tmp/fakeproj")

    def test_slash_command_defaults_to_session_project(self):
        """Without a project the command runs in the session's project."""
        tui, _ = make_tui()
        captured = {}

        def fake_start(text, system=None, restore=None):
            captured["text"] = text
            captured["restore"] = restore

        with mock.patch.object(tui, "_start_agent", side_effect=fake_start):
            tui._handle_slash("/init")
        self.assertIn("/tmp/fakeproj", captured["text"])
        self.assertIsNone(captured["restore"])

    def test_slash_init_hides_planexit_in_plan_mode(self):
        """/init runs with every tool except PlanExit: the tool is
        hidden for the run (sub-agents share the session registry, so
        they are covered too) and restored when the run finishes."""
        tui, _ = make_tui()
        tui.session.switch_to_plan()
        self.assertIsNotNone(tui.session.registry.get("PlanExit"))

        captured = {}

        def fake_start(text, system=None, restore=None):
            captured["restore"] = restore
            self.assertIsNone(tui.session.registry.get("PlanExit"))
            restore()  # simulate the run finishing

        with mock.patch.object(tui, "_start_agent", side_effect=fake_start):
            tui._handle_slash("/init")
        self.assertIsNotNone(captured["restore"])
        self.assertIsNotNone(tui.session.registry.get("PlanExit"))

    def test_slash_command_keeps_planexit_for_custom(self):
        """Custom commands (/explain) may use all tools, incl. PlanExit."""
        tui, _ = make_tui()
        tui.session.switch_to_plan()
        captured = {}

        def fake_start(text, system=None, restore=None):
            captured["restore"] = restore
            self.assertIsNotNone(tui.session.registry.get("PlanExit"))

        with mock.patch.object(tui, "_start_agent", side_effect=fake_start):
            tui._handle_slash("/explain client.py")
        self.assertIsNone(captured["restore"])
        self.assertIsNotNone(tui.session.registry.get("PlanExit"))

    def test_explain_requires_target(self):
        tui, buf = make_tui()
        with mock.patch.object(tui, "_start_agent") as start:
            tui._handle_slash("/explain")
        start.assert_not_called()
        self.assertIn("needs a target", buf.getvalue())

    def test_unknown_slash_command(self):
        tui, buf = make_tui()
        self.assertFalse(tui._handle_slash("/bogus"))
        self.assertIn("unknown command", buf.getvalue())

    def test_help_lists_command_slashes(self):
        tui, buf = make_tui()
        tui._handle_slash("/help")
        out = buf.getvalue()
        for s in ("/init", "/review", "/explain"):
            self.assertIn(s, out)
        # bracket usage text must not be swallowed by rich markup
        for s in (
            "/init [project] [--extra TEXT]       create/update AGENTS.md",
            "/review [project] [commit|branch|PR] review code changes",
            "/explain [project] [target]",
            "/restore [path | title | --latest | latest]   restore a saved session",
        ):
            self.assertIn(s, out)

    # ------------------------------------------------------------------
    # remaining slash commands
    # ------------------------------------------------------------------
    def test_exit_slash(self):
        tui, _ = make_tui()
        self.assertTrue(tui._handle_slash("/exit"))

    def test_plan_and_build_slashes(self):
        tui, buf = make_tui()
        self.assertFalse(tui._handle_slash("/plan"))
        self.assertEqual(tui.session.plan_mode.mode.value, "plan")
        self.assertIn("Plan mode", buf.getvalue())
        self.assertFalse(tui._handle_slash("/build"))
        self.assertEqual(tui.session.plan_mode.mode.value, "build")
        self.assertIn("Build mode", buf.getvalue())

    def test_save_slash(self):
        tui, buf = make_tui()
        with mock.patch.object(tui.session.store, "save", return_value="/tmp/x.md"):
            self.assertFalse(tui._handle_slash("/save"))
        self.assertIn("saved: /tmp/x.md", buf.getvalue())

    def test_compact_and_summary_slashes_dispatch(self):
        tui, _ = make_tui()
        with (
            mock.patch.object(tui, "_run_compact") as c,
            mock.patch.object(tui, "_run_summary") as s,
        ):
            tui._handle_slash("/compact")
            tui._handle_slash("/summary")
        c.assert_called_once_with()
        s.assert_called_once_with()

    def test_sessions_and_restore_slashes_dispatch(self):
        tui, _ = make_tui()
        with (
            mock.patch.object(tui, "_run_sessions") as s,
            mock.patch.object(tui, "_run_restore") as r,
        ):
            tui._handle_slash("/sessions")
            tui._handle_slash("/restore foo.md")
        s.assert_called_once_with()
        r.assert_called_once_with("foo.md")

    def test_split_args_unbalanced_quote_falls_back(self):
        """An unterminated quote falls back to whitespace splitting."""
        tui, _ = make_tui()
        self.assertEqual(tui._split_args('unterminated "quote'), ["unterminated", '"quote'])

    def test_command_args_init_invalid_returns_none(self):
        """/init with a non-project token after the project is invalid."""
        tui, _ = make_tui()
        self.assertEqual(tui._command_args("init", "proj --extra"), ("proj", None))
        self.assertEqual(tui._command_args("init", "a b"), (None, None))

    def test_run_slash_command_unknown(self):
        """A slash command with no registered SessionCommand is reported."""
        tui, buf = make_tui()
        with mock.patch("python_agent_harness.tui.commands.find_command", return_value=None):
            tui._run_slash_command("bogus", "")
        self.assertIn("unknown command: /bogus", buf.getvalue())

    def test_planexit_restore_idempotent_with_prev_restore(self):
        """The planexit-restore wrapper undoes the project borrow first
        and ignores repeat invocations."""
        tui, _ = make_tui()
        with tempfile.TemporaryDirectory() as d:
            tui.session.switch_to_plan()
            seen = []

            def fake_start(text, system=None, restore=None):
                restore()
                restore()  # second call must be a no-op
                seen.append(tui.session.project_dir)

            with mock.patch.object(tui, "_start_agent", side_effect=fake_start):
                tui._handle_slash(f"/init {d}")
        self.assertEqual(tui.session.project_dir, "/tmp/fakeproj")
        self.assertEqual(seen, ["/tmp/fakeproj"])
        self.assertIsNotNone(tui.session.registry.get("PlanExit"))

    # ------------------------------------------------------------------
    # /sessions
    # ------------------------------------------------------------------
    def test_run_sessions_empty(self):
        tui, buf = make_tui()
        with mock.patch(
            "python_agent_harness.tui.commands.SessionPersistence.list_sessions", return_value=[]
        ):
            tui._run_sessions()
        self.assertIn("no saved sessions", buf.getvalue())

    def test_run_sessions_lists_metadata(self):
        tui, buf = make_tui()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "my session_250101120000.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    "**user**: hello\n\n;; Local Variables:\n"
                    ";; gptel-model: gpt-4\n"
                    ";; python-agent-harness--project-dir: /tmp/p\n"
                    ";; End:\n"
                )
            with mock.patch(
                "python_agent_harness.tui.commands.SessionPersistence.list_sessions",
                return_value=[path],
            ):
                tui._run_sessions()
        out = buf.getvalue()
        self.assertIn("my session_250101120000.md", out)
        self.assertIn("gpt-4", out)
        self.assertIn("/tmp/p", out)

    def test_run_sessions_skips_unreadable_files(self):
        tui, buf = make_tui()
        with mock.patch(
            "python_agent_harness.tui.commands.SessionPersistence.list_sessions",
            return_value=["/nonexistent/session.md"],
        ):
            tui._run_sessions()  # must not raise
        self.assertEqual(buf.getvalue(), "")

    # ------------------------------------------------------------------
    # /restore paths
    # ------------------------------------------------------------------
    def test_restore_no_session_found(self):
        """/restore with nothing to restore prints the yellow hint."""
        tui, buf = make_tui()
        with mock.patch(
            "python_agent_harness.tui.commands.SessionPersistence.latest_session",
            return_value=None,
        ):
            tui._run_restore("")
        self.assertIn("no session found", buf.getvalue())

    def test_restore_latest_session(self):
        """/restore --latest and /restore latest both load the most recent
        session file (same code branch, two accepted spellings)."""
        tui, buf = make_tui()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "session.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write("**user**: hello\n\n**assistant**: hi")
            for arg in ("--latest", "latest"):
                buf.truncate(0)
                with mock.patch(
                    "python_agent_harness.tui.commands.SessionPersistence.latest_session",
                    return_value=path,
                ) as latest:
                    tui._run_restore(arg)
                latest.assert_called_once_with()
                out = buf.getvalue()
                self.assertIn("restored:", out)
                self.assertIn("session.md", out)
                self.assertEqual([m.text() for m in tui.session.last_messages], ["hello", "hi"])

    def test_restore_resolved_path_not_a_file(self):
        """A resolved path that is not a file reports an error."""
        tui, buf = make_tui()
        with mock.patch(
            "python_agent_harness.tui.commands.SessionPersistence.latest_session",
            return_value="/nonexistent/session.md",
        ):
            tui._run_restore("--latest")
        self.assertIn("file not found", buf.getvalue())

    def test_restore_unreadable_file(self):
        tui, buf = make_tui()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "session.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write("**user**: hello")
            with mock.patch("builtins.open", side_effect=OSError("denied")):
                tui._run_restore(path)
        self.assertIn("cannot read", buf.getvalue())

    def test_restore_by_title_match(self):
        """A non-path /restore arg matches session filenames/titles, and a
        title-bearing filename sets the store title."""
        tui, buf = make_tui()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "my session_250101120000.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write("**user**: hello\n\n**assistant**: hi")
            with mock.patch(
                "python_agent_harness.tui.commands.SessionPersistence.list_sessions",
                return_value=[path],
            ):
                tui._run_restore("MY SESSION")
        self.assertIn("restored:", buf.getvalue())
        self.assertEqual(tui.session.store.title, "my session")

    def test_find_session_by_title(self):
        """Title lookup: exact basename, .md-less, substring and
        derived-title matches; unmatched queries return None."""
        with tempfile.TemporaryDirectory() as d:
            dash = os.path.join(d, "fix-bugs_250101000000.md")
            spaced = os.path.join(d, "Add feature_250101000001.md")
            for f in (dash, spaced):
                open(f, "w", encoding="utf-8").close()
            files = [dash, spaced]
            with mock.patch(
                "python_agent_harness.tui.commands.SessionPersistence.list_sessions",
                return_value=files,
            ):
                # exact basename match (with and without .md)
                self.assertEqual(
                    Tui._find_session_by_title("Add feature_250101000001.md"),
                    spaced,
                )
                self.assertEqual(Tui._find_session_by_title("add feature_250101000001"), spaced)
                # filename substring match
                self.assertEqual(Tui._find_session_by_title("fix-bugs"), dash)
                self.assertEqual(Tui._find_session_by_title("feature"), spaced)
                # derived-title match (dashes -> spaces)
                self.assertEqual(Tui._find_session_by_title("fix bugs"), dash)
                self.assertIsNone(Tui._find_session_by_title("nothing here"))

    # ------------------------------------------------------------------
    # /model paths
    # ------------------------------------------------------------------
    def test_model_list_always_includes_default(self):
        """The numbered list always shows ``default`` first followed by
        every profile, so the count stays stable across switches and
        the original model stays selectable."""
        tui, buf = make_tui()
        tui.session.model_profiles = {
            "deepseek": {"model": "deepseek-chat"},
            "glm": {"model": "glm-5.2"},
        }
        tui.session.model = "glm-5.2"  # current model IS a profile
        names = tui._model_list_names()
        self.assertEqual(names, ["default", "deepseek", "glm"])
        # current model NOT in profiles -> same stable list
        tui.session.model = "elsewhere-model"
        names = tui._model_list_names()
        self.assertEqual(names, ["default", "deepseek", "glm"])

    def test_model_numbered_selection_matches_list(self):
        """``/model N`` picks the same entry the numbered list showed:
        ``1`` is always ``default``, then profiles in order."""
        tui, buf = make_tui()
        profiles = {
            "deepseek": {"model": "deepseek-chat"},
            "glm": {"model": "glm-5.2"},
        }
        tui.session.model_profiles = dict(profiles)
        tui.session.llm_settings = {"model": "gpt-5-mini", "base_url": "https://default"}
        tui.session.model = "glm-5.2"  # current IS a profile (index 3)
        with mock.patch(
            "python_agent_harness.tui.commands.config.load_models_config", return_value=profiles
        ):
            # 1 == default: switches back to the original model
            with mock.patch.object(tui, "_model_switch_by_name") as switch:
                tui._run_model_command("1")
            switch.assert_called_once_with("default")
            buf.truncate(0)
            with mock.patch.object(tui, "_model_switch_by_name") as switch:
                tui._run_model_command("2")
            switch.assert_called_once_with("deepseek")
            buf.truncate(0)
            with mock.patch.object(tui, "_model_switch_by_name") as switch:
                tui._run_model_command("3")
            switch.assert_called_once_with("glm")
            buf.truncate(0)
            # back on the default model: 1 == default -> already using
            tui.session.model = "gpt-5-mini"
            with mock.patch.object(tui, "_model_switch_by_name") as switch:
                tui._run_model_command("1")
            switch.assert_not_called()
            self.assertIn("Already using this model", buf.getvalue())
            buf.truncate(0)
            with mock.patch.object(tui, "_model_switch_by_name") as switch:
                tui._run_model_command("2")
            switch.assert_called_once_with("deepseek")

    def test_model_interactive_selection_can_switch_back_to_default(self):
        """The interactive selection can switch back to the original
        default model after switching to a profile."""
        tui, buf = make_tui()
        profiles = {"deepseek": {"model": "deepseek-chat"}}
        tui.session.model_profiles = dict(profiles)
        tui.session.llm_settings = {"model": "gpt-5-mini", "base_url": "https://default"}
        tui.session.model = "deepseek-chat"
        with (
            mock.patch("builtins.input", return_value="1"),
            mock.patch(
                "python_agent_harness.tui.commands.config.load_models_config", return_value=profiles
            ),
            mock.patch.object(tui, "_model_switch_by_name") as switch,
        ):
            tui._run_model_command("")
        switch.assert_called_once_with("default")

    def test_model_switch_by_name(self):
        """``/model <name>`` switches via the session."""
        tui, buf = make_tui()
        profiles = {"deepseek": {"model": "deepseek-chat"}}
        tui.session.model_profiles = dict(profiles)
        with (
            mock.patch(
                "python_agent_harness.tui.commands.config.load_models_config", return_value=profiles
            ),
            mock.patch.object(tui.session, "switch_model", return_value=(True, "switched")) as sw,
        ):
            tui._run_model_command("deepseek")
        sw.assert_called_once_with("deepseek")
        self.assertIn("switched", buf.getvalue())

    def test_model_reloads_profiles_from_config_each_call(self):
        """``/model`` re-reads the config file on every call, so a
        profile added mid-session shows up and is switchable without
        restarting, and ``default`` stays available with none set."""
        tui, buf = make_tui()
        # no profiles configured: default is still listed
        with (
            mock.patch(
                "python_agent_harness.tui.commands.config.load_models_config", return_value={}
            ),
            mock.patch("builtins.input", return_value=""),
        ):
            tui._run_model_command("")
        self.assertIn("default", buf.getvalue())
        self.assertIn("none configured", buf.getvalue())
        buf.truncate(0)
        # profile added to the config file mid-session -> visible next call
        new_profiles = {"new": {"model": "new-model", "base_url": "https://new/v1"}}
        with (
            mock.patch(
                "python_agent_harness.tui.commands.config.load_models_config",
                return_value=new_profiles,
            ),
            mock.patch("builtins.input", return_value=""),
        ):
            tui._run_model_command("")
        self.assertIn("new", buf.getvalue())
        buf.truncate(0)
        # and switchable by name immediately
        with mock.patch(
            "python_agent_harness.tui.commands.config.load_models_config", return_value=new_profiles
        ):
            tui._run_model_command("new")
        self.assertEqual(tui.session.model, "new-model")
        self.assertEqual(tui.session.client.base_url, "https://new/v1")


if __name__ == "__main__":
    unittest.main()
