"""Extra Controller tests: submit with only failed @file references,
the attachment path of _build_user_message, and the staleness/restore
guards of _run_worker."""

from __future__ import annotations

import os
import tempfile
import threading
import unittest
from unittest import mock

from python_agent_harness.core.models import Message, TextPart
from python_agent_harness.entry.controller import Controller


class FakeSession:
    """Minimal session surface used by the tested Controller paths."""

    def __init__(self) -> None:
        self.project_dir = "/tmp"
        self.run_generation = 0
        self.last_messages: list[Message] = []
        self.cancel_event = threading.Event()
        self.llm_settings: dict = {}
        self.model_profiles: dict = {}
        self.client = mock.Mock()
        self.model = "test-model"
        self.system_prompt = "you are a fake session"
        self.store = mock.Mock()

    def clear_todos(self) -> None:
        pass

    def notify(self, kind: str, data=None) -> None:
        pass

    def log(self, msg: str) -> None:
        pass


class TestSubmitWithoutContent(unittest.TestCase):
    def test_only_failed_references_returns_none(self):
        """The guard fires when parsing left nothing to send: no
        attachments, empty cleaned text, errors present (the parser's
        contract that drives this branch)."""
        from python_agent_harness.io.attachments import AttachmentError

        ctrl = Controller(FakeSession())
        err = AttachmentError("missing.txt", "file not found: missing.txt")
        with mock.patch(
            "python_agent_harness.entry.controller.parse_at_references",
            return_value=("", [], [err]),
        ):
            handle = ctrl.submit("@missing.txt")
        self.assertIsNone(handle)

    def test_failed_reference_keeps_token_in_text(self):
        """A failed @file reference is kept verbatim in the message (so
        the user sees what failed) and the error rides on the handle."""
        ctrl = Controller(FakeSession())
        msg, display, errors, _ = ctrl._build_user_message("@missing.txt")
        self.assertIn("@missing.txt", display)
        self.assertEqual(len(errors), 1)
        self.assertIn("missing.txt", errors[0].path)


class TestBuildUserMessage(unittest.TestCase):
    def test_attachment_only_produces_multimodal_message(self):
        ctrl = Controller(FakeSession())
        with tempfile.TemporaryDirectory() as d:
            ctrl.session.project_dir = d
            with open(os.path.join(d, "notes.txt"), "w") as f:
                f.write("content\n")
            msg, display, errors, warnings = ctrl._build_user_message("@notes.txt")
        self.assertEqual(errors, [])
        self.assertIsInstance(msg.content, list)
        self.assertIsInstance(msg.content[0], TextPart)
        # text attachments keep the path as display text (context for
        # the inlined content); images would show a placeholder instead
        self.assertEqual(display, "notes.txt")

    def test_text_plus_attachment_keeps_text_and_parts(self):
        ctrl = Controller(FakeSession())
        with tempfile.TemporaryDirectory() as d:
            ctrl.session.project_dir = d
            with open(os.path.join(d, "data.csv"), "w") as f:
                f.write("a,b\n")
            msg, display, _, _ = ctrl._build_user_message("look at @data.csv please")
        self.assertIsInstance(msg.content, list)
        assert isinstance(msg.content, list)
        self.assertIn("look at", msg.content[0].text)
        self.assertTrue(display.startswith("look at"))

    def test_message_input_passes_through(self):
        ctrl = Controller(FakeSession())
        original = Message(role="user", content="already built")
        msg, display, errors, warnings = ctrl._build_user_message(original)
        self.assertIs(msg, original)
        self.assertEqual(display, "already built")
        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])

    def test_message_with_empty_text_uses_attachment_placeholder(self):
        ctrl = Controller(FakeSession())
        empty = Message(role="user", content="")
        _, display, _, _ = ctrl._build_user_message(empty)
        self.assertEqual(display, "(attachment)")


class TestRunWorkerGuards(unittest.TestCase):
    def _ctrl(self, session: FakeSession) -> Controller:
        return Controller(session)

    def test_skips_history_adoption_when_no_messages(self):
        """last_messages empty -> the salvaged-history adoption is
        skipped (the arc that guards the assignment)."""
        session = FakeSession()
        session.run_generation = 1
        session.last_messages = []
        ctrl = self._ctrl(session)
        with mock.patch("python_agent_harness.entry.controller.run_agent_loop") as loop:
            ctrl._run_worker(Message(role="user", content="hi"), seq=1, system=None, restore=None)
        loop.assert_called_once()  # no exception path
        self.assertEqual(session.run_generation, 1)
        self.assertEqual(len(ctrl.conversation_history), 1)
        self.assertEqual(ctrl.conversation_history[0].role, "user")  # no adoption

    def test_restore_callback_runs_when_current(self):
        session = FakeSession()
        session.run_generation = 1
        session.last_messages = [Message(role="assistant", content="done")]
        ctrl = self._ctrl(session)
        restore = mock.Mock()
        with mock.patch("python_agent_harness.entry.controller.run_agent_loop") as loop:
            ctrl._run_worker(
                Message(role="user", content="hi"), seq=1, system=None, restore=restore
            )
        loop.assert_called_once()  # no exception path
        restore.assert_called_once()
        # adoption ran: the (mocked) loop's last_messages replaced history
        self.assertEqual(ctrl.conversation_history[0].role, "assistant")

    def test_stale_worker_skips_adoption_and_restore(self):
        """A worker whose seq no longer matches must not adopt history
        nor run the restore callback."""
        session = FakeSession()
        session.run_generation = 2  # a newer run started
        session.last_messages = [Message(role="assistant", content="stale")]
        ctrl = self._ctrl(session)
        restore = mock.Mock()
        with mock.patch("python_agent_harness.entry.controller.run_agent_loop") as loop:
            ctrl._run_worker(
                Message(role="user", content="hi"), seq=1, system=None, restore=restore
            )
        loop.assert_called_once()  # no exception path
        restore.assert_not_called()
        self.assertEqual(len(ctrl.conversation_history), 1)  # only the user msg
        self.assertEqual(ctrl.conversation_history[0].role, "user")  # no adoption


if __name__ == "__main__":
    unittest.main()
