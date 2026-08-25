"""TUI input tests: SlashCompleter, Tab/Shift+Tab key bindings and
_read_multiline prompt handling."""

import os
import sys
import tempfile
import unittest
import unittest.mock as mock

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import plan_cleanup  # noqa: F401,E402  (side-effect: auto-remove /tmp plan dirs)
from tui_test_utils import make_tui


class TestTuiInput(unittest.TestCase):
    # ------------------------------------------------------------------
    # completer
    # ------------------------------------------------------------------
    def test_completer_slash_commands(self):
        from prompt_toolkit.document import Document

        from python_agent_harness.tui import SlashCompleter

        c = SlashCompleter(get_project_dir=lambda: "/tmp/fakeproj")
        completions = list(c.get_completions(Document(text="/ini", cursor_position=4), None))
        names = [x.text for x in completions]
        self.assertIn("/init", names)
        self.assertNotIn("/plan", names)
        completions = list(c.get_completions(Document(text="/", cursor_position=1), None))
        names = [x.text for x in completions]
        for cmd in ("/plan", "/build", "/init", "/review", "/exit"):
            self.assertIn(cmd, names)

    def test_completer_tilde_paths(self):
        """~/wor + Tab must complete to ~/workspace (the user's case),
        bare ~ completes to ~/, and mid-sentence ~-tokens complete too."""
        from prompt_toolkit.document import Document

        from python_agent_harness.tui import SlashCompleter

        with tempfile.TemporaryDirectory() as d:
            os.mkdir(os.path.join(d, "workspace"))
            os.mkdir(os.path.join(d, "workbench"))
            with mock.patch.dict(os.environ, {"HOME": d}):
                c = SlashCompleter(get_project_dir=lambda: "/tmp/fakeproj")
                completions = list(
                    c.get_completions(Document(text="~/wor", cursor_position=5), None)
                )
                names = [x.text for x in completions]
                self.assertIn("kspace/", names)  # workspace
                self.assertIn("kbench/", names)  # workbench
                # bare ~ -> the trailing slash only (home dir itself)
                completions = list(c.get_completions(Document(text="~", cursor_position=1), None))
                self.assertEqual([x.text for x in completions], ["/"])
                # mid-sentence token completes
                completions = list(
                    c.get_completions(Document(text="see ~/wor", cursor_position=9), None)
                )
                self.assertIn("kspace/", [x.text for x in completions])

    def test_completer_plain_text_no_completion(self):
        from prompt_toolkit.document import Document

        from python_agent_harness.tui import SlashCompleter

        c = SlashCompleter(get_project_dir=lambda: "/tmp/fakeproj")
        for text in ("hello", "fix the /init bug", ""):
            completions = list(
                c.get_completions(Document(text=text, cursor_position=len(text)), None)
            )
            self.assertEqual(completions, [], f"unexpected completions for {text!r}")

    def test_completer_directories(self):
        from prompt_toolkit.document import Document

        from python_agent_harness.tui import SlashCompleter

        with tempfile.TemporaryDirectory() as d:
            os.mkdir(os.path.join(d, "alpha"))
            os.mkdir(os.path.join(d, "beta"))
            open(os.path.join(d, "file.txt"), "w").close()
            open(os.path.join(d, "alpha", "inner.py"), "w").close()
            c = SlashCompleter(get_project_dir=lambda: d)
            completions = list(c.get_completions(Document(text="/init ", cursor_position=6), None))
            names = [x.text for x in completions]
            self.assertIn("alpha/", names)  # directories get a trailing slash
            self.assertIn("beta/", names)
            self.assertIn("file.txt", names)  # files complete too (e.g. /explain)
            # partial dir prefix: only the suffix is inserted at the cursor
            completions = list(
                c.get_completions(Document(text="/init al", cursor_position=8), None)
            )
            self.assertEqual([x.text for x in completions], ["pha/"])
            # empty arg lists the project dir's own contents, not its siblings
            completions = list(c.get_completions(Document(text="/init ", cursor_position=6), None))
            self.assertIn("alpha/", [x.text for x in completions])
            # trailing slash drills into the subdirectory
            completions = list(
                c.get_completions(Document(text="/init alpha/", cursor_position=12), None)
            )
            self.assertIn("inner.py", [x.text for x in completions])

    def test_completer_absolute_path_fallback(self):
        """A /-token matching no slash command completes as an absolute
        path (and yields nothing when nothing matches)."""
        from prompt_toolkit.document import Document

        from python_agent_harness.tui import SlashCompleter

        c = SlashCompleter(get_project_dir=lambda: "/tmp/fakeproj")
        completions = list(
            c.get_completions(Document(text="/zzzz-no-such", cursor_position=13), None)
        )
        self.assertEqual(completions, [])

    def test_completer_unlistable_directory_no_crash(self):
        """A path whose directory cannot be listed yields no completions
        instead of raising."""
        from prompt_toolkit.document import Document

        from python_agent_harness.tui import SlashCompleter

        c = SlashCompleter(get_project_dir=lambda: "/tmp/fakeproj")
        completions = list(
            c.get_completions(Document(text="/init /no/such/dir/", cursor_position=16), None)
        )
        self.assertEqual(completions, [])

    def test_tab_key_binding_completes(self):
        """Tab (c-i) must trigger completion end-to-end, and Shift+Tab
        must cycle backwards through the completion menu."""
        import asyncio

        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput

        from python_agent_harness.tui import SlashCompleter, _make_prompt_session

        async def run(text: str, keys: str) -> str:
            with tempfile.TemporaryDirectory() as d, create_pipe_input() as inp:
                c = SlashCompleter(get_project_dir=lambda: "/tmp/fakeproj")
                s = _make_prompt_session(
                    FileHistory(os.path.join(d, "hist")),
                    c,
                    input=inp,
                    output=DummyOutput(),
                )
                task = asyncio.ensure_future(s.prompt_async("> "))
                await asyncio.sleep(0.1)
                inp.send_text(text)
                await asyncio.sleep(0.2)
                inp.send_text(keys)
                await asyncio.sleep(0.3)
                inp.send_text("\x1b\r")
                return await asyncio.wait_for(task, 5)

        self.assertEqual(asyncio.run(run("/ini", "\t")), "/init")

    def test_tab_burst_input_completes(self):
        """Text and Tab arriving in a single input burst must still
        complete (regression: complete_while_typing's background task
        used to create the completion state first, so the Tab-triggered
        task bailed out without inserting)."""
        import asyncio

        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput

        from python_agent_harness.tui import SlashCompleter, _make_prompt_session

        async def run(burst: str) -> str:
            with tempfile.TemporaryDirectory() as d, create_pipe_input() as inp:
                c = SlashCompleter(get_project_dir=lambda: "/tmp/fakeproj")
                s = _make_prompt_session(
                    FileHistory(os.path.join(d, "hist")),
                    c,
                    input=inp,
                    output=DummyOutput(),
                )
                task = asyncio.ensure_future(s.prompt_async("> "))
                await asyncio.sleep(0.1)
                inp.send_text(burst)  # text + Tab in one chunk
                await asyncio.sleep(0.5)
                inp.send_text("\x1b\r")
                return await asyncio.wait_for(task, 5)

        self.assertEqual(asyncio.run(run("/ini\t")), "/init")

    def test_shift_tab_key_binding(self):
        """Shift+Tab (s-tab) must exist as a key binding handler."""
        from prompt_toolkit.key_binding.key_bindings import KeyBindings

        from python_agent_harness.tui import _make_key_bindings

        kb = _make_key_bindings()
        self.assertIsInstance(kb, KeyBindings)
        handlers = {b.keys: b.handler for b in kb.bindings}
        self.assertIn(("c-i",), handlers)  # Tab
        self.assertIn(("s-tab",), handlers)  # Shift+Tab

    # ------------------------------------------------------------------
    # key-binding handlers (Tab / Shift+Tab with an open completion menu)
    # ------------------------------------------------------------------
    def _kb_handlers(self):
        from python_agent_harness.tui import _make_key_bindings

        return {b.keys: b.handler for b in _make_key_bindings().bindings}

    def test_complete_handler_with_menu_cycles_forward(self):
        """Tab while a completion menu is open cycles to the next entry."""
        buffer = mock.Mock()
        buffer.complete_state = object()  # menu open
        self._kb_handlers()[("c-i",)](mock.Mock(current_buffer=buffer))
        buffer.complete_next.assert_called_once_with()
        buffer.start_completion.assert_not_called()

    def test_complete_handler_starts_menu(self):
        """Tab with no menu open starts completion with the common part."""
        buffer = mock.Mock()
        buffer.complete_state = None
        self._kb_handlers()[("c-i",)](mock.Mock(current_buffer=buffer))
        buffer.start_completion.assert_called_once_with(insert_common_part=True)

    def test_complete_backward_handler_cycles(self):
        """Shift+Tab while a menu is open cycles to the previous entry."""
        buffer = mock.Mock()
        buffer.complete_state = object()
        self._kb_handlers()[("s-tab",)](mock.Mock(current_buffer=buffer))
        buffer.complete_previous.assert_called_once_with()
        buffer.start_completion.assert_not_called()

    def test_complete_backward_handler_starts_menu(self):
        """Shift+Tab with no menu open starts completion selecting the first."""
        buffer = mock.Mock()
        buffer.complete_state = None
        self._kb_handlers()[("s-tab",)](mock.Mock(current_buffer=buffer))
        buffer.start_completion.assert_called_once_with(select_first=True)

    # ------------------------------------------------------------------
    # _read_multiline
    # ------------------------------------------------------------------
    def test_read_multiline_eof_quits(self):
        tui, _ = make_tui()
        with mock.patch.object(tui.prompt_session, "prompt", side_effect=EOFError):
            self.assertIsNone(tui._read_multiline())

    def test_read_multiline_interrupt_cancels_input(self):
        tui, buf = make_tui()
        with mock.patch.object(tui.prompt_session, "prompt", side_effect=KeyboardInterrupt):
            self.assertEqual(tui._read_multiline(), "")
        self.assertIn("input cancelled", buf.getvalue())

    def test_read_multiline_uses_styled_prompt(self):
        """The input prompt is a styled FormattedText carrying the short
        model name (no org prefix), not a bare '> '."""
        from prompt_toolkit.formatted_text import FormattedText

        tui, _ = make_tui()
        tui.session.model = "deepseek-ai/deepseek-flash-v4"
        with mock.patch.object(tui.prompt_session, "prompt", return_value="hello") as m:
            self.assertEqual(tui._read_multiline(), "hello")
        prompt = m.call_args.args[0]
        self.assertIsInstance(prompt, FormattedText)
        plain = "".join(text for _, text in prompt)
        self.assertIn("deepseek-flash-v4", plain)  # short model name
        self.assertNotIn("deepseek-ai/", plain)  # org prefix stripped
        self.assertTrue(plain.endswith("> "))

    def test_read_multiline_uses_styled_prompt_with_title(self):
        """When a session title is available, the input prompt shows the
        short model name plus the dimmed title in parentheses, truncated
        to 20 chars."""
        from prompt_toolkit.formatted_text import FormattedText

        tui, _ = make_tui()
        tui.session.model = "deepseek-ai/deepseek-flash-v4"
        tui.session.store.title = "A very long session title that exceeds twenty chars"
        with mock.patch.object(tui.prompt_session, "prompt", return_value="hello") as m:
            self.assertEqual(tui._read_multiline(), "hello")
        prompt = m.call_args.args[0]
        self.assertIsInstance(prompt, FormattedText)
        plain = "".join(text for _, text in prompt)
        self.assertIn("deepseek-flash-v4", plain)  # short model name
        self.assertNotIn("deepseek-ai/", plain)  # org prefix stripped
        # title present, truncated to 20 chars, wrapped in parens
        self.assertIn("(A very long session )", plain)
        self.assertNotIn("exceeds twenty chars", plain)
        self.assertTrue(plain.endswith("> "))
        # the title fragment is rendered dim
        styles = [style for style, _ in prompt]
        self.assertIn("dim", styles)


if __name__ == "__main__":
    unittest.main()
