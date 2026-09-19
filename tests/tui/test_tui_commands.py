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
from python_agent_harness.persistence import find_session_by_title


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
            self.assertIn("AGENTS.md", captured["text"].text())
            self.assertIn("Create or update", captured["system"])

            tui._handle_slash("/review main")
            self.assertIn("Review the requested code changes", captured["text"].text())
            self.assertIn("code reviewer", captured["system"])
            self.assertIn("main", captured["system"])  # $ARGUMENTS substituted

            tui._handle_slash("/explain client.py")
            self.assertIn("instructions", captured["text"].text())  # custom kickoff
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
        self.assertIn("NEW /explain request: client.py", captured["text"].text())
        self.assertIn("background context", captured["text"].text())
        self.assertIn(
            "Proceed with the task described in your instructions.", captured["text"].text()
        )

        # empty conversation: kickoff stays the plain generic message
        tui.conversation_history = []
        tui.session.last_messages = []
        with mock.patch.object(tui, "_start_agent", side_effect=fake_start):
            tui._handle_slash("/explain client.py")
        self.assertEqual(
            captured["text"].text().strip(),
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
        self.assertIn("NEW /review request", captured["text"].text())
        self.assertNotIn("NEW /review request:", captured["text"].text())

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
        self.assertIn("/tmp/fakeproj", captured["text"].text())
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

    def test_arbitrary_custom_command_dispatches(self):
        """Any custom command (not just /explain) is routed through
        _run_slash_command: a temp commands dir with a fresh .md file
        must dispatch, not fall through to 'unknown command'."""
        import tempfile
        from pathlib import Path

        import python_agent_harness.prompts as prompts_pkg
        from python_agent_harness.prompts import core as prompts_mod
        from python_agent_harness.session import commands as commands_mod

        tui, buf = make_tui()
        captured = {}

        def fake_start(text, system=None, restore=None):
            captured["text"] = text
            captured["system"] = system

        with tempfile.TemporaryDirectory() as d:
            prompts_dir = Path(d) / "prompts"
            cmds_dir = prompts_dir / "commands"
            cmds_dir.mkdir(parents=True)
            (cmds_dir / "custom_test.md").write_text(
                "You are a custom command test.", encoding="utf-8"
            )
            with (
                mock.patch.object(commands_mod, "COMMANDS_DIR", cmds_dir),
                mock.patch.object(commands_mod, "PROMPTS_DIR", prompts_dir),
                # read_prompt_file resolves against prompts.PROMPTS_DIR,
                # so the temp commands dir must be visible there too.
                mock.patch.object(prompts_mod, "PROMPTS_DIR", prompts_dir),
                mock.patch.object(prompts_pkg, "PROMPTS_DIR", prompts_dir),
                mock.patch.object(tui, "_start_agent", side_effect=fake_start),
            ):
                self.assertFalse(tui._handle_slash("/custom-test"))
        self.assertIn("custom-test", captured["text"].text())
        self.assertIn("You are a custom command test.", captured["system"])
        self.assertNotIn("unknown command", buf.getvalue())

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
                    ";; python-agent-harness--model: gpt-4\n"
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
        # the console may wrap the line, so allow whitespace/newline
        self.assertRegex(out, r"1\s+messages")

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

    # ------------------------------------------------------------------
    # /restore: agent + model restoration
    # ------------------------------------------------------------------
    def test_restore_applies_saved_agent(self):
        """A session saved under a custom agent switches back to it on
        restore; the summary line shows the agent name."""
        tui, buf = make_tui()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "session.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    "**user**: hi\n\n**assistant**: hello\n\n;; Local Variables:\n"
                    ";; python-agent-harness--model: fake\n"
                    ";; python-agent-harness--agent: 'reviewer'\n"
                    ";; End:\n"
                )
            tui._run_restore(path)
        out = buf.getvalue()
        self.assertIn("switched to reviewer", out)
        self.assertIn("agent=reviewer", out)
        self.assertIn("code reviewer", tui.session.system_prompt)
        self.assertEqual(tui.session.store.agent, "reviewer")

    def test_restore_without_agent_metadata_keeps_current(self):
        """Old session files (no agent metadata) restore cleanly: the
        active agent stays untouched and no warning is printed."""
        tui, buf = make_tui()
        tui.session.store.agent = "reviewer"
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "old.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    "**user**: hi\n\n;; Local Variables:\n"
                    ";; python-agent-harness--model: fake\n"
                    ";; End:\n"
                )
            tui._run_restore(path)
        out = buf.getvalue()
        self.assertNotIn("switched", out)
        self.assertNotIn("warning", out)
        self.assertEqual(tui.session.store.agent, "reviewer")

    def test_restore_unknown_agent_warns_and_keeps_current(self):
        """An agent name that no longer exists produces a warning, leaves
        the current agent active, and the history is still restored."""
        tui, buf = make_tui()
        before_prompt = tui.session.system_prompt
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "session.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    "**user**: hi\n\n;; Local Variables:\n"
                    ";; python-agent-harness--model: fake\n"
                    ";; python-agent-harness--agent: 'no-such-agent'\n"
                    ";; End:\n"
                )
            tui._run_restore(path)
        out = buf.getvalue()
        self.assertIn("warning: unknown agent", out)
        self.assertEqual(tui.session.system_prompt, before_prompt)
        # conversation history restored regardless
        self.assertEqual([m.text() for m in tui.session.last_messages], ["hi"])
        self.assertIn("restored:", out)

    def test_restore_model_via_matching_profile(self):
        """A saved raw model name matching a profile's model field switches
        through the profile (fixing base_url etc. together with the name)."""
        tui, buf = make_tui()
        tui.session.llm_settings = {"model": "start-model", "stream": True}
        tui.session.model_profiles = {"deepseek": {"model": "ds-chat", "base_url": "https://ds/v1"}}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "session.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    "**user**: hi\n\n;; Local Variables:\n"
                    ";; python-agent-harness--model: 'ds-chat'\n"
                    ";; End:\n"
                )
            tui._run_restore(path)
        self.assertEqual(tui.session.model, "ds-chat")
        self.assertEqual(tui.session.client.model, "ds-chat")
        self.assertEqual(tui.session.client.base_url, "https://ds/v1")
        self.assertIn("switched to deepseek", buf.getvalue())

    def test_restore_default_model_via_pseudo_profile(self):
        """A saved model equal to the session-start default restores the
        original llm settings through the ``default`` pseudo-profile,
        even after the current session drifted to another profile."""
        tui, buf = make_tui()
        tui.session.llm_settings = {"model": "start-model", "stream": True}
        tui.session.model_profiles = {"other": {"model": "other-model", "base_url": "https://o/v1"}}
        tui.session.switch_model("other")
        self.assertEqual(tui.session.model, "other-model")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "session.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    "**user**: hi\n\n;; Local Variables:\n"
                    ";; python-agent-harness--model: 'start-model'\n"
                    ";; End:\n"
                )
            tui._run_restore(path)
        self.assertEqual(tui.session.model, "start-model")
        self.assertIn("switched to default", buf.getvalue())

    def test_restore_model_without_match_keeps_current(self):
        """A saved model with no profile-name or profile-model match warns
        and keeps the current model; the history is still restored."""
        tui, buf = make_tui()
        tui.session.llm_settings = {"model": "start-model", "stream": True}
        tui.session.model_profiles = {}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "session.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    "**user**: hi\n\n;; Local Variables:\n"
                    ";; python-agent-harness--model: 'mystery-model'\n"
                    ";; End:\n"
                )
            tui._run_restore(path)
        out = buf.getvalue()
        self.assertIn("no matching profile", out)
        self.assertEqual(tui.session.model, "fake")  # make_tui's model
        self.assertEqual([m.text() for m in tui.session.last_messages], ["hi"])

    def test_restore_saved_agent_and_model_together(self):
        """Full save/restore round trip: agent and model from the saved
        metadata are both applied on top of the restored history."""
        tui, buf = make_tui()
        tui.session.llm_settings = {"model": "start-model", "stream": True}
        tui.session.model_profiles = {"glm": {"model": "glm-5.2", "base_url": "https://glm/v1"}}
        # simulate a saved session: save with active agent + profile model
        tui.session.switch_agent("reviewer")
        tui.session.switch_model("glm")
        saved_text = tui.session._conversation_text(tui.session.last_messages)
        saved_meta = tui.session.store.metadata_block()
        self.assertIn("python-agent-harness--agent: 'reviewer'", saved_meta)
        # fresh session, restore from the saved text
        tui2, buf2 = make_tui()
        tui2.session.llm_settings = {"model": "start-model", "stream": True}
        tui2.session.model_profiles = dict(tui.session.model_profiles)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "session.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(saved_text + "\n\n" + saved_meta + "\n")
            tui2._run_restore(path)
        self.assertIn("code reviewer", tui2.session.system_prompt)
        self.assertEqual(tui2.session.model, "glm-5.2")
        self.assertEqual(tui2.session.client.base_url, "https://glm/v1")
        out = buf2.getvalue()
        self.assertIn("switched to reviewer", out)
        self.assertIn("switched to glm", out)
        self.assertIn("agent=reviewer", out)

    def test_restore_drops_system_and_tool_blocks(self):
        """``**system**:`` and ``**tool**:`` blocks in a saved body are
        dropped: a restored system message would duplicate the live
        prompt, a restored tool message would be API-invalid."""
        tui, buf = make_tui()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "session.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    "**system**: stale prompt\n\n"
                    "**user**: hi\n\n"
                    "**tool**: stale tool output\n\n"
                    "**assistant**: hello"
                )
            tui._run_restore(path)
        roles = [m.role for m in tui.session.last_messages]
        self.assertEqual(roles, ["user", "assistant"])

    def test_restore_switches_input_history_to_saved_project(self):
        """/restore of a session saved under another project points the
        prompt session's Up/Down recall at that project's history file
        (session + default buffer), not the startup project's."""
        from python_agent_harness.tui.input import _history_path

        tui, buf = make_tui()
        startup_project = str(tui._controller.project_dir)
        saved_project = "/some/other/project"
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "session.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    "**user**: hi\n\n;; Local Variables:\n"
                    f";; python-agent-harness--project-dir: {saved_project!r}\n"
                    ";; End:\n"
                )
            tui._run_restore(path)
        expected = _history_path(saved_project)
        self.assertEqual(tui.prompt_session.history.filename, expected)
        self.assertEqual(tui.prompt_session.default_buffer.history.filename, expected)
        self.assertNotEqual(expected, _history_path(startup_project))

    def test_restore_without_project_metadata_keeps_input_history(self):
        """A session file without project metadata (old format) leaves
        the prompt session's input history untouched."""
        tui, buf = make_tui()
        before = tui.prompt_session.history.filename
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "session.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write("**user**: hi\n\n;; Local Variables:\n;; End:\n")
            tui._run_restore(path)
        self.assertEqual(tui.prompt_session.history.filename, before)
        self.assertEqual(tui.prompt_session.default_buffer.history.filename, before)

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
                    find_session_by_title("Add feature_250101000001.md"),
                    spaced,
                )
                self.assertEqual(find_session_by_title("add feature_250101000001"), spaced)
                # filename substring match
                self.assertEqual(find_session_by_title("fix-bugs"), dash)
                self.assertEqual(find_session_by_title("feature"), spaced)
                # derived-title match (dashes -> spaces)
                self.assertEqual(find_session_by_title("fix bugs"), dash)
                self.assertIsNone(find_session_by_title("nothing here"))

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


class TestTuiCommandsExtra(unittest.TestCase):
    """Remaining branches: /model + /agent interactive flows, /help with
    no custom commands, kickoff attachments, image placeholders,
    _run_with_status terminal modes, agent listing/switching, and the
    /restore warning paths."""

    # ------------------------------------------------------------------
    # slash dispatch
    # ------------------------------------------------------------------
    def test_slash_dispatch_routes_model_and_agent(self):
        tui, _ = make_tui()
        with (
            mock.patch.object(tui, "_run_model_command") as model,
            mock.patch.object(tui, "_run_agent_command") as agent,
        ):
            self.assertFalse(tui._handle_slash("/model deepseek"))
            model.assert_called_once_with("deepseek")
            self.assertFalse(tui._handle_slash("/agent planner"))
            agent.assert_called_once_with("planner")

    def test_help_without_custom_commands(self):
        """With no custom commands the help omits the custom block —
        exercising the empty-custom branch of the listing."""
        tui, buf = make_tui()
        with mock.patch("python_agent_harness.commands.load_custom_commands", return_value=[]):
            tui._handle_slash("/help")
        out = buf.getvalue()
        self.assertIn("/sessions", out)
        self.assertIn("Ctrl-D or /exit quits.", out)

    # ------------------------------------------------------------------
    # kickoff attachments + image placeholders
    # ------------------------------------------------------------------
    def test_kickoff_attachments_and_reference_errors(self):
        """@file references in the kickoff become message parts; failed
        references are reported but the run still proceeds."""
        from python_agent_harness.models import TextPart

        tui, buf = make_tui()
        tui.conversation_history = []
        tui.session.last_messages = []
        captured = {}

        def fake_start(text, system=None, restore=None):
            captured["text"] = text

        attached = TextPart(text="[attached file contents]")
        bad = mock.Mock()
        bad.path = "missing.png"
        bad.message = "no such file"
        with (
            mock.patch(
                "python_agent_harness.tui.commands.parse_at_references",
                return_value=("do the task", [mock.Mock(part=attached)], [bad]),
            ),
            mock.patch.object(tui, "_start_agent", side_effect=fake_start),
        ):
            tui._run_slash_command("init", "")
        self.assertIn("@missing.png: no such file", buf.getvalue())
        content = captured["text"].content
        self.assertIsInstance(content, list)
        self.assertEqual(content[0].text, "do the task")
        self.assertIs(content[-1], attached)

    def test_conversation_text_image_placeholder(self):
        """Messages with image parts get a placeholder line recording the
        image count and sources (so a saved session shows what was
        attached)."""
        from python_agent_harness.models import ImagePart, TextPart

        tui, _ = make_tui()
        tui.session.last_messages = [
            Message(
                role="user",
                content=[
                    TextPart(text="look at this"),
                    ImagePart(data=b"x", path="/tmp/shot.png"),
                    ImagePart(url="https://example.com/i.png"),
                ],
            ),
        ]
        text = tui._conversation_text()
        self.assertIn("2 image", text)
        self.assertIn("/tmp/shot.png", text)
        self.assertIn("https://example.com/i.png", text)
        self.assertIn("look at this", text)

    # ------------------------------------------------------------------
    # _run_with_status: terminal modes
    # ------------------------------------------------------------------
    def test_run_with_status_dumb_terminal_loop(self):
        """In a dumb terminal the status bar is printed per poll instead
        of using rich's Live region."""
        import threading
        import time as _time

        from rich.console import Console

        tui, buf = make_tui()
        done = threading.Event()

        def worker() -> None:
            _time.sleep(0.15)
            done.set()

        with mock.patch.object(
            Console, "is_dumb_terminal", new_callable=mock.PropertyMock, return_value=True
        ):
            tui._run_with_status(worker, status_text=" ⏳ working", cancel_message="cancelled")
        self.assertTrue(done.is_set())
        self.assertFalse(tui.agent_running)

    def test_run_with_status_keyboard_interrupt_reports_cancel(self):
        import time as _time

        tui, buf = make_tui()

        def worker() -> None:
            _time.sleep(0.3)

        with mock.patch.object(tui._data_event, "wait", side_effect=KeyboardInterrupt):
            tui._run_with_status(worker, status_text="x", cancel_message="aborted!")
        self.assertIn("aborted!", buf.getvalue())
        self.assertFalse(tui.agent_running)

    def test_refresh_model_profiles_reports_bad_config(self):
        tui, buf = make_tui()
        tui.session.model_profiles = {"keep": {"model": "m"}}
        with mock.patch(
            "python_agent_harness.tui.commands.config.load_models_config",
            side_effect=ValueError("bad config"),
        ):
            tui._refresh_model_profiles()
        self.assertIn("bad config", buf.getvalue())
        self.assertEqual(tui.session.model_profiles, {"keep": {"model": "m"}})

    def test_model_switch_by_name_failure_reported(self):
        tui, buf = make_tui()
        with mock.patch.object(
            tui.session, "switch_model", return_value=(False, "no such profile")
        ):
            tui._model_switch_by_name("nope")
        self.assertIn("no such profile", buf.getvalue())

    # ------------------------------------------------------------------
    # /model interactive flows
    # ------------------------------------------------------------------
    def _set_default_model(self, tui) -> None:
        profiles: dict = {}
        tui.session.model_profiles = dict(profiles)
        tui.session.llm_settings = {"model": "gpt-5-mini", "base_url": "https://default"}
        tui.session.model = "gpt-5-mini"

    def test_model_interactive_already_default(self):
        tui, buf = make_tui()
        self._set_default_model(tui)
        with (
            mock.patch(
                "python_agent_harness.tui.commands.config.load_models_config", return_value={}
            ),
            mock.patch("builtins.input", return_value="1"),
        ):
            tui._run_model_command("")
        self.assertIn("Already using this model", buf.getvalue())

    def test_model_interactive_invalid_number(self):
        tui, buf = make_tui()
        self._set_default_model(tui)
        with (
            mock.patch(
                "python_agent_harness.tui.commands.config.load_models_config", return_value={}
            ),
            mock.patch("builtins.input", return_value="99"),
        ):
            tui._run_model_command("")
        self.assertIn("Invalid selection: 99", buf.getvalue())

    def test_model_interactive_name_switches(self):
        tui, buf = make_tui()
        self._set_default_model(tui)
        with (
            mock.patch(
                "python_agent_harness.tui.commands.config.load_models_config", return_value={}
            ),
            mock.patch("builtins.input", return_value="some-model"),
            mock.patch.object(tui, "_model_switch_by_name") as switch,
        ):
            tui._run_model_command("")
        switch.assert_called_once_with("some-model")

    def test_model_interactive_eof_is_silent(self):
        tui, buf = make_tui()
        self._set_default_model(tui)
        with (
            mock.patch(
                "python_agent_harness.tui.commands.config.load_models_config", return_value={}
            ),
            mock.patch("builtins.input", side_effect=EOFError),
        ):
            tui._run_model_command("")  # must not raise
        self.assertNotIn("cancelled", buf.getvalue())

    def test_model_interactive_keyboard_interrupt_cancels(self):
        tui, buf = make_tui()
        self._set_default_model(tui)
        with (
            mock.patch(
                "python_agent_harness.tui.commands.config.load_models_config", return_value={}
            ),
            mock.patch("builtins.input", side_effect=KeyboardInterrupt),
        ):
            tui._run_model_command("")
        self.assertIn("cancelled", buf.getvalue())

    def test_model_numeric_arg_out_of_range(self):
        tui, buf = make_tui()
        self._set_default_model(tui)
        with mock.patch(
            "python_agent_harness.tui.commands.config.load_models_config", return_value={}
        ):
            tui._run_model_command("99")
        self.assertIn("Invalid selection: 99", buf.getvalue())

    # ------------------------------------------------------------------
    # /summary fallback
    # ------------------------------------------------------------------
    def test_summary_with_contentless_last_message_prints_status(self):
        tui, buf = make_tui()
        with mock.patch.object(
            tui.session, "summarize_conversation", return_value="Summary appended."
        ):
            tui.session.last_messages = [Message(role="assistant", content="")]
            tui._run_summary()
        self.assertIn("Summary appended.", buf.getvalue())

    # ------------------------------------------------------------------
    # /agent command
    # ------------------------------------------------------------------
    def test_refresh_agent_profiles_discovers(self):
        tui, _ = make_tui()
        with mock.patch(
            "python_agent_harness.tui.commands.discover_agents",
            return_value={"planner": "/p/planner.md"},
        ):
            tui._refresh_agent_profiles()
        self.assertEqual(tui._discovered_agents, {"planner": "/p/planner.md"})
        self.assertEqual(tui._agent_list_names(), ["default", "planner"])

    def test_agent_switch_by_name_success_and_failure(self):
        tui, buf = make_tui()
        with mock.patch.object(tui.session, "switch_agent", return_value=(True, "agent ok")):
            tui._agent_switch_by_name("planner")
        self.assertIn("agent ok", buf.getvalue())
        buf.truncate(0)
        with mock.patch.object(tui.session, "switch_agent", return_value=(False, "no agent")):
            tui._agent_switch_by_name("nope")
        self.assertIn("no agent", buf.getvalue())

    def test_agent_command_lists_without_agents(self):
        tui, buf = make_tui()
        with (
            mock.patch("python_agent_harness.tui.commands.discover_agents", return_value={}),
            mock.patch("builtins.input", return_value=""),
        ):
            tui._run_agent_command("")
        out = buf.getvalue()
        self.assertIn("Available agent profiles", out)
        self.assertIn("none found", out)

    def test_agent_command_interactive_number(self):
        tui, buf = make_tui()
        with (
            mock.patch(
                "python_agent_harness.tui.commands.discover_agents",
                return_value={"planner": "/p/planner.md"},
            ),
            mock.patch("builtins.input", return_value="2"),
            mock.patch.object(tui, "_agent_switch_by_name") as switch,
        ):
            tui._run_agent_command("")
        switch.assert_called_once_with("planner")

    def test_agent_command_interactive_invalid_number(self):
        tui, buf = make_tui()
        with (
            mock.patch("python_agent_harness.tui.commands.discover_agents", return_value={}),
            mock.patch("builtins.input", return_value="9"),
        ):
            tui._run_agent_command("")
        self.assertIn("Invalid selection: 9", buf.getvalue())

    def test_agent_command_interactive_name(self):
        tui, buf = make_tui()
        with (
            mock.patch("python_agent_harness.tui.commands.discover_agents", return_value={}),
            mock.patch("builtins.input", return_value="planner"),
            mock.patch.object(tui, "_agent_switch_by_name") as switch,
        ):
            tui._run_agent_command("")
        switch.assert_called_once_with("planner")

    def test_agent_command_interactive_eof_and_interrupt(self):
        tui, buf = make_tui()
        with (
            mock.patch("python_agent_harness.tui.commands.discover_agents", return_value={}),
            mock.patch("builtins.input", side_effect=EOFError),
        ):
            tui._run_agent_command("")  # silent
        with (
            mock.patch("python_agent_harness.tui.commands.discover_agents", return_value={}),
            mock.patch("builtins.input", side_effect=KeyboardInterrupt),
        ):
            tui._run_agent_command("")
        self.assertIn("cancelled", buf.getvalue())

    def test_agent_command_numeric_and_named_arg(self):
        tui, buf = make_tui()
        with (
            mock.patch(
                "python_agent_harness.tui.commands.discover_agents",
                return_value={"planner": "/p/planner.md"},
            ),
            mock.patch.object(tui, "_agent_switch_by_name") as switch,
        ):
            tui._run_agent_command("1")
        switch.assert_called_once_with("default")
        switch.reset_mock()
        with mock.patch(
            "python_agent_harness.tui.commands.discover_agents",
            return_value={"planner": "/p/planner.md"},
        ):
            tui._run_agent_command("99")
        self.assertIn("Invalid selection: 99", buf.getvalue())
        buf.truncate(0)
        with (
            mock.patch(
                "python_agent_harness.tui.commands.discover_agents",
                return_value={"planner": "/p/planner.md"},
            ),
            mock.patch.object(tui, "_agent_switch_by_name") as switch,
        ):
            tui._run_agent_command("planner")
        switch.assert_called_once_with("planner")

    # ------------------------------------------------------------------
    # /restore warning paths
    # ------------------------------------------------------------------
    def test_restore_bad_round_times_metadata_tolerated(self):
        tui, buf = make_tui()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "session.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    "**user**: hello\n\n;; Local Variables:\n"
                    ";; python-agent-harness--round-times: not-a-number\n"
                    ";; End:\n"
                )
            tui._run_restore(path)
        self.assertIn("restored:", buf.getvalue())
        self.assertEqual(tui._round_times, [])

    def test_restore_agent_switch_exception_warns(self):
        tui, buf = make_tui()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "session.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    "**user**: hello\n\n;; Local Variables:\n"
                    ";; python-agent-harness--agent: planner\n"
                    ";; End:\n"
                )
            with mock.patch.object(tui.session, "switch_agent", side_effect=RuntimeError("boom")):
                tui._run_restore(path)
        self.assertIn("could not restore agent: boom", buf.getvalue())

    def test_restore_model_switch_exception_warns(self):
        tui, buf = make_tui()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "session.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    "**user**: hello\n\n;; Local Variables:\n"
                    ";; python-agent-harness--model: other-model\n"
                    ";; End:\n"
                )
            with mock.patch.object(tui.session, "switch_model", side_effect=RuntimeError("boom")):
                tui._run_restore(path)
        self.assertIn("could not restore model: boom", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
