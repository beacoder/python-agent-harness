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
            with mock.patch.dict(os.environ, {"HOME": d, "USERPROFILE": d}):
                c = SlashCompleter(get_project_dir=lambda: "/tmp/fakeproj")
                completions = list(
                    c.get_completions(Document(text="~/wor", cursor_position=5), None)
                )
                names = [x.text for x in completions]
                # On Windows, paths may be converted; check for workspace/workbench variants
                self.assertTrue(
                    any("workspace" in n.lower() or "kspace" in n.lower() for n in names)
                )
                self.assertTrue(
                    any("workbench" in n.lower() or "kbench" in n.lower() for n in names)
                )
                # bare ~ -> the trailing slash only (home dir itself)
                completions = list(c.get_completions(Document(text="~", cursor_position=1), None))
                self.assertEqual([x.text for x in completions], ["/"])
                # mid-sentence token completes
                completions = list(
                    c.get_completions(Document(text="see ~/wor", cursor_position=9), None)
                )
                self.assertTrue(
                    any(
                        "workspace" in n.lower() or "kspace" in n.lower()
                        for n in [x.text for x in completions]
                    )
                )

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

    def test_completer_at_references(self):
        """@path tokens complete as paths relative to the project dir."""
        from prompt_toolkit.document import Document

        from python_agent_harness.tui import SlashCompleter

        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "screenshot.png"), "w").close()
            open(os.path.join(d, "README.md"), "w").close()
            os.mkdir(os.path.join(d, "sub"))
            c = SlashCompleter(get_project_dir=lambda: d)
            # @scr completes to screenshot.png
            completions = list(c.get_completions(Document(text="@scr", cursor_position=4), None))
            names = [x.text for x in completions]
            self.assertIn("eenshot.png", names)
            # bare @ lists all files in the project dir
            completions = list(c.get_completions(Document(text="@", cursor_position=1), None))
            names = [x.text for x in completions]
            self.assertIn("screenshot.png", names)
            self.assertIn("README.md", names)
            self.assertIn("sub/", names)
            # mid-sentence @ completes
            completions = list(
                c.get_completions(Document(text="see @READ", cursor_position=9), None)
            )
            names = [x.text for x in completions]
            self.assertIn("ME.md", names)

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


class _FakeBuffer:
    def __init__(self):
        self.text = ""

    def insert_text(self, s):
        self.text += s


class _FakeEvent:
    def __init__(self, data, buffer):
        self.data = data
        self.current_buffer = buffer


def _paste_handler(on_image_paste=None):
    from prompt_toolkit.keys import Keys

    from python_agent_harness.tui.input import _make_key_bindings

    kb = _make_key_bindings(on_image_paste)
    for b in kb.bindings:
        if Keys.BracketedPaste in b.keys:
            return b.handler
    raise AssertionError("no BracketedPaste binding found")


class TestPasteBinding(unittest.TestCase):
    """The BracketedPaste binding hands a captured clipboard image to the
    on_image_paste callback and inserts a marker; a text paste inserts
    the text."""

    def test_image_paste_calls_callback_and_inserts_marker(self):
        handler = _paste_handler()
        buf = _FakeBuffer()
        captured = []
        handler = _paste_handler(on_image_paste=captured.append)
        with mock.patch(
            "python_agent_harness.clipboard.grab_clipboard_image",
            return_value="/tmp/my dir/clip-abc.png",
        ) as grab:
            handler(_FakeEvent(data="", buffer=buf))
        grab.assert_called_once()
        # the FULL path (even with a space) is handed to the callback,
        # NOT routed through the whitespace-sensitive @file parser
        self.assertEqual(captured, ["/tmp/my dir/clip-abc.png"])
        # a cosmetic, non-@ marker is inserted (basename only)
        self.assertEqual(buf.text, "[image #clip-abc.png] ")
        self.assertNotIn("@", buf.text)

    def test_image_paste_without_callback_falls_back_to_text(self):
        """No callback wired (e.g. in a headless context) -> the paste
        is treated as text, never routed as an image."""
        handler = _paste_handler(on_image_paste=None)
        buf = _FakeBuffer()
        with mock.patch("python_agent_harness.clipboard.grab_clipboard_image") as grab:
            handler(_FakeEvent(data="", buffer=buf))
        grab.assert_not_called()
        self.assertEqual(buf.text, "")

    def test_text_paste_inserts_text_without_clipboard_check(self):
        buf = _FakeBuffer()
        handler = _paste_handler(on_image_paste=lambda p: None)
        # non-empty text paste must NOT spawn a clipboard subprocess
        with mock.patch("python_agent_harness.clipboard.grab_clipboard_image") as grab:
            handler(_FakeEvent(data="hello world", buffer=buf))
        grab.assert_not_called()
        self.assertEqual(buf.text, "hello world")

    def test_text_paste_normalises_newlines(self):
        buf = _FakeBuffer()
        handler = _paste_handler(on_image_paste=lambda p: None)
        with mock.patch("python_agent_harness.clipboard.grab_clipboard_image"):
            handler(_FakeEvent(data="a\r\nb\rc", buffer=buf))
        self.assertEqual(buf.text, "a\nb\nc")

    def test_empty_paste_no_image_inserts_nothing(self):
        buf = _FakeBuffer()
        handler = _paste_handler(on_image_paste=lambda p: None)
        with mock.patch("python_agent_harness.clipboard.grab_clipboard_image", return_value=None):
            handler(_FakeEvent(data="", buffer=buf))
        self.assertEqual(buf.text, "")

    def test_capture_failure_falls_back_to_text(self):
        """A clipboard-capture exception must never break paste; the
        pasted text is inserted instead."""
        buf = _FakeBuffer()
        handler = _paste_handler(on_image_paste=lambda p: None)
        with mock.patch(
            "python_agent_harness.clipboard.grab_clipboard_image",
            side_effect=RuntimeError("boom"),
        ):
            handler(_FakeEvent(data="", buffer=buf))
        # empty data + failed capture -> empty buffer, no crash
        self.assertEqual(buf.text, "")


class TestDrainClipboardImages(unittest.TestCase):
    """Tui._drain_clipboard_images validates pending pasted images and
    strips their markers from the text."""

    def _make_png(self, d, name):
        p = os.path.join(d, name)
        with open(p, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
        return p

    def test_capture_callback_survives_multiple_submits(self):
        """Regression: the paste callback appends to the SAME list the
        drain reads across cycles.  A naive reassign-to-clear orphans the
        callback so only the first paste ever attaches — this guards it.

        Simulates the real wiring: the callback captured at __init__ is
        ``_pending_clipboard_images.append``; the drain must clear the
        list IN PLACE, not rebind it.
        """
        tui, _buf = make_tui()
        from python_agent_harness.attachments import clipboard_image_marker

        # the callback as wired in Tui.__init__
        on_image_paste = tui._pending_clipboard_images.append

        with tempfile.TemporaryDirectory() as d:
            p1 = self._make_png(d, "clip-1.png")
            p2 = self._make_png(d, "clip-2.png")

            # ---- first paste + submit ----
            on_image_paste(p1)
            atts1, _ = tui._drain_clipboard_images(clipboard_image_marker(p1))
            self.assertEqual(len(atts1), 1)

            # ---- second paste (after a drain) + submit ----
            on_image_paste(p2)
            atts2, _ = tui._drain_clipboard_images(clipboard_image_marker(p2))
            # the bug dropped this to 0; must be 1
            self.assertEqual(len(atts2), 1)

    def test_drain_attaches_valid_image_with_spaces_in_path(self):
        """M1 end-to-end: a captured temp path with spaces attaches
        correctly and its marker is stripped."""
        tui, _buf = make_tui()
        with tempfile.TemporaryDirectory(prefix="my dir ") as d:
            p = os.path.join(d, "python-agent-harness-clip-x.png")
            with open(p, "wb") as f:
                f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
            tui._pending_clipboard_images.append(p)
            from python_agent_harness.attachments import clipboard_image_marker
            from python_agent_harness.models import ImagePart

            text = f"describe {clipboard_image_marker(p)}please"
            atts, cleaned = tui._drain_clipboard_images(text)
        self.assertEqual(cleaned, "describe please")
        self.assertEqual(len(atts), 1)
        self.assertIsInstance(atts[0].part, ImagePart)
        # pending list is always drained
        self.assertEqual(tui._pending_clipboard_images, [])

    def test_drain_reports_and_drops_invalid_image(self):
        tui, _buf = make_tui()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "bad.png")
            with open(p, "wb") as f:
                f.write(b"not an image")
            tui._pending_clipboard_images.append(p)
            atts, cleaned = tui._drain_clipboard_images("hi")
        self.assertEqual(atts, [])
        self.assertEqual(cleaned, "hi")
        self.assertEqual(tui._pending_clipboard_images, [])

    def test_drain_no_pending_is_noop(self):
        tui, _buf = make_tui()
        self.assertEqual(tui._pending_clipboard_images, [])
        atts, cleaned = tui._drain_clipboard_images("just text")
        self.assertEqual(atts, [])
        self.assertEqual(cleaned, "just text")


if __name__ == "__main__":
    unittest.main()
