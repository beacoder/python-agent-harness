"""TUI question tests: numbered/keyed choice resolution, _ask_question_blocking,
_ask_sync, _ui_ask and _ui_confirm."""

import os
import sys
import unittest
import unittest.mock as mock

sys.path.insert(0, os.path.dirname(__file__))

import plan_cleanup  # noqa: F401,E402  (side-effect: auto-remove /tmp plan dirs)
from tui_test_utils import make_tui

from python_agent_harness.tui import UiQuestion, _resolve_keyed_choice, _resolve_numbered_choice


class TestTuiQuestions(unittest.TestCase):
    # ------------------------------------------------------------------
    # question selection (number keys)
    # ------------------------------------------------------------------
    def test_numbered_choice_resolution(self):
        """Bare numbers map to option labels; everything else passes
        through untouched (free-text answers, out-of-range, empty)."""
        options = ["foo bar", "baz", "qux"]
        self.assertEqual(_resolve_numbered_choice("1", options), "foo bar")
        self.assertEqual(_resolve_numbered_choice("2", options), "baz")
        self.assertEqual(_resolve_numbered_choice("3", options), "qux")
        self.assertEqual(_resolve_numbered_choice("1,3", options), "foo bar, qux")
        self.assertEqual(_resolve_numbered_choice("2, custom", options), "baz, custom")
        self.assertEqual(_resolve_numbered_choice("custom", options), "custom")
        self.assertEqual(_resolve_numbered_choice("0", options), "0")
        self.assertEqual(_resolve_numbered_choice("9", options), "9")
        self.assertEqual(_resolve_numbered_choice("", options), "")
        self.assertEqual(_resolve_numbered_choice("1", []), "1")
        self.assertEqual(_resolve_numbered_choice("1", ["only"]), "only")

    def test_ask_question_short_options_numbered_too(self):
        """Single-letter options (y/n/a/d) also render as a numbered
        list — numbers apply to ALL option lists now — and a number
        resolves to the matching option."""
        tui, buf = make_tui()
        q = UiQuestion("Proceed?", options=["y", "n"])
        tui.question = q
        with mock.patch.object(tui.prompt_session, "prompt", return_value="2") as m:
            tui._ask_question_blocking()
        self.assertEqual(q.answer, "n")
        m.assert_called_once_with("> ", multiline=False)
        out = buf.getvalue()
        self.assertIn("1) y", out)
        self.assertIn("2) n", out)

    def test_ask_question_custom_answer_passthrough(self):
        """Free-text answers (not numbers) are returned verbatim."""
        tui, _ = make_tui()
        q = UiQuestion("Pick one", options=["long option a", "long option b"])
        tui.question = q
        with mock.patch.object(tui.prompt_session, "prompt", return_value="something else"):
            tui._ask_question_blocking()
        self.assertEqual(q.answer, "something else")

    def test_ask_question_multiple_numbers(self):
        """Comma-separated numbers select several options (multiple)."""
        tui, _ = make_tui()
        q = UiQuestion(
            "Pick several",
            multiple=True,
            options=["first choice", "second choice", "third choice"],
        )
        tui.question = q
        with mock.patch.object(tui.prompt_session, "prompt", return_value="1,3"):
            tui._ask_question_blocking()
        self.assertEqual(q.answer, "first choice, third choice")

    # ------------------------------------------------------------------
    # PlanExit confirmation (y/n keyed list, like Question but keys)
    # ------------------------------------------------------------------
    def test_keyed_choice_resolution(self):
        """Keys map to the matching option label; non-keys pass through."""
        options = ["Yes, switch to build agent", "No, keep refining the plan"]
        keys = ["y", "n"]
        self.assertEqual(_resolve_keyed_choice("y", options, keys), options[0])
        self.assertEqual(_resolve_keyed_choice("n", options, keys), options[1])
        self.assertEqual(_resolve_keyed_choice("Y", options, keys), options[0])
        self.assertEqual(_resolve_keyed_choice("y, custom", options, keys), f"{options[0]}, custom")
        self.assertEqual(_resolve_keyed_choice("custom", options, keys), "custom")
        self.assertEqual(_resolve_keyed_choice("", options, keys), "")
        self.assertEqual(_resolve_keyed_choice("y", [], []), "y")

    def test_ask_question_keyed_list_renders_and_resolves(self):
        """A keyed choice renders as a list (y) label / n) label) with a
        hint line, and a typed key resolves to the option label."""
        tui, buf = make_tui()
        q = UiQuestion(
            "Approve plan?",
            options=["Yes, switch to build agent", "No, keep refining the plan"],
            keys=["y", "n"],
            custom=False,
        )
        tui.question = q
        with mock.patch.object(tui.prompt_session, "prompt", return_value="n"):
            tui._ask_question_blocking()
        out = buf.getvalue()
        self.assertIn("Approve plan?", out)
        self.assertIn("y) Yes, switch to build agent", out)
        self.assertIn("n) No, keep refining the plan", out)
        self.assertIn("Enter a key", out)
        self.assertEqual(q.answer, "No, keep refining the plan")

    def test_ui_confirm_accepts_y_n_and_legacy_yes(self):
        """_ui_confirm approves on y/yes, rejects on n; it renders a
        y/n keyed choice list (not a bare prompt)."""
        from python_agent_harness import config

        tui, _ = make_tui()
        for raw, expected in (
            ("y", True),
            ("n", False),
            ("yes", True),
            ("a", True),
            ("1", True),
            ("", False),
        ):
            with mock.patch.object(tui, "_ask_sync", return_value=raw) as ask:
                self.assertEqual(tui._ui_confirm("Switch to build?"), expected)
            q = ask.call_args[0][0]
            self.assertEqual(q.options, list(config.PLAN_EXIT_OPTIONS))
            self.assertEqual(q.keys, ["y", "n"])
            self.assertFalse(q.custom)

    # ------------------------------------------------------------------
    # cancel-aware question wait + auto-save error surfacing
    # ------------------------------------------------------------------
    def test_ask_sync_unblocks_on_cancel(self):
        """A Ctrl-C (session cancel) while a question is pending must
        unblock the worker's wait promptly with an empty answer."""
        import threading
        import time

        tui, _ = make_tui()
        q = UiQuestion("Pick one", options=["a", "b"])
        results = {}
        worker = threading.Thread(target=lambda: results.update(r=tui._ask_sync(q)))
        worker.start()
        time.sleep(0.2)
        tui.session.cancel()  # sets cancel_event
        worker.join(timeout=2)
        self.assertFalse(worker.is_alive(), "question wait wedged by cancel")
        self.assertEqual(results["r"], "")

    def test_ask_sync_returns_answer_when_not_cancelled(self):
        tui, _ = make_tui()
        q = UiQuestion("Pick one", options=["a", "b"])
        q.answer = "b"
        q.event.set()
        self.assertEqual(tui._ask_sync(q), "b")

    # ------------------------------------------------------------------
    # _ui_ask (Question tool)
    # ------------------------------------------------------------------
    def test_ui_ask_single_question(self):
        """A single Question returns one 'prompt' = 'answer' line."""
        tui, _ = make_tui()
        with mock.patch.object(tui, "_ask_sync", return_value="42"):
            result = tui._ui_ask([{"question": "How many?", "options": ["one", "two"]}])
        self.assertEqual(result, '"How many?" = "42"')

    def test_ui_ask_multiple_questions(self):
        tui, _ = make_tui()
        with mock.patch.object(tui, "_ask_sync", side_effect=["x", "y"]):
            result = tui._ui_ask([{"question": "Q1"}, {"question": "Q2"}])
        self.assertEqual(result, '"Q1" = "x"\n"Q2" = "y"')

    def test_ui_ask_multiple_joins_and_cleans_answers(self):
        """Multiple-select answers are joined, dropping empty parts."""
        tui, _ = make_tui()
        with mock.patch.object(tui, "_ask_sync", return_value="a, , b"):
            result = tui._ui_ask([{"question": "Pick", "multiple": True, "options": ["a", "b"]}])
        self.assertEqual(result, '"Pick" = "a, b"')

    def test_ui_ask_no_questions_returns_unanswered(self):
        tui, _ = make_tui()
        self.assertEqual(tui._ui_ask([]), "Unanswered")

    # ------------------------------------------------------------------
    # question prompt variants
    # ------------------------------------------------------------------
    def test_ask_question_keyed_multiple_custom_hints(self):
        """A keyed list with multiple+custom shows both hint extensions
        and resolves comma-separated keys."""
        tui, buf = make_tui()
        q = UiQuestion(
            "Approve?",
            multiple=True,
            options=["Yes, switch", "No, refine"],
            keys=["y", "n"],
            custom=True,
        )
        tui.question = q
        with mock.patch.object(tui.prompt_session, "prompt", return_value="y,n"):
            tui._ask_question_blocking()
        self.assertEqual(q.answer, "Yes, switch, No, refine")
        out = buf.getvalue()
        self.assertIn("Enter keys, comma-separated", out)
        self.assertIn("or type your own answer", out)

    def test_ask_question_plain_prompt(self):
        """A question without options uses a bare 'prompt > ' line."""
        tui, _ = make_tui()
        q = UiQuestion("What is your name?")
        tui.question = q
        with mock.patch.object(tui.prompt_session, "prompt", return_value="Ada") as m:
            tui._ask_question_blocking()
        m.assert_called_once_with("What is your name? > ", multiline=False)
        self.assertEqual(q.answer, "Ada")

    def test_ask_question_eof_returns_empty(self):
        """Ctrl-D/Ctrl-C at a question prompt answers with an empty string."""
        tui, _ = make_tui()
        q = UiQuestion("Pick", options=["a", "b"])
        tui.question = q
        with mock.patch.object(tui.prompt_session, "prompt", side_effect=EOFError):
            tui._ask_question_blocking()
        self.assertEqual(q.answer, "")
        self.assertIsNone(tui.question)


if __name__ == "__main__":
    unittest.main()
