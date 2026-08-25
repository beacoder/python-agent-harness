"""TUI scrollback-dump and session save/restore parsing tests."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import plan_cleanup  # noqa: F401,E402  (side-effect: auto-remove /tmp plan dirs)
from tui_test_utils import make_tui

from python_agent_harness.models import Message
from python_agent_harness.tui import Tui


class TestTuiDump(unittest.TestCase):
    def test_dump_conversation_full_history(self):
        """After a run, the full conversation is printed as plain lines
        (not Live frames) so it lands in the terminal scrollback —
        including rows the visible frame budget would have dropped."""
        tui, buf = make_tui()
        tui.session.last_messages = [
            Message(role="user", content=f"message number {i}") for i in range(120)
        ]
        tui._dump_conversation()
        out = buf.getvalue()
        self.assertIn("message number 0", out)  # oldest rows included
        self.assertIn("message number 119", out)  # newest rows included
        self.assertIn("full conversation", out)

    def test_dump_conversation_unlimited(self):
        """No line cap: a huge conversation is dumped in full, oldest
        and newest rows alike."""
        tui, buf = make_tui()
        tui.session.last_messages = [Message(role="user", content=f"m{i}") for i in range(3000)]
        tui._dump_conversation()
        out = buf.getvalue()
        self.assertIn("m0", out)  # oldest row included
        self.assertIn("m2999", out)  # newest row included
        self.assertNotIn("omitted", out)

    def test_dump_conversation_empty_noop(self):
        """No messages → no dump, no separator line."""
        tui, buf = make_tui()
        tui.session.last_messages = []
        tui._dump_conversation()
        self.assertEqual(buf.getvalue(), "")

    def test_dump_conversation_full_long_reply(self):
        """The scrollback dump must show long assistant replies in FULL.

        The live panel tail-caps long messages to the newest lines
        (regression: the dump reused those same capped rows, so the
        head of a long summary was never readable anywhere in the TUI
        — only its tail, prefixed with a "…" marker).
        """
        tui, buf = make_tui()
        body = "\n".join(f"summary line {i}" for i in range(40))
        tui.session.last_messages = [Message(role="assistant", content=body)]
        tui._dump_conversation()
        out = buf.getvalue()
        self.assertIn("summary line 0", out)  # head visible
        self.assertIn("summary line 39", out)  # tail visible
        self.assertNotIn("…\n", out)  # no tail-cut marker
        self.assertNotIn("more lines", out)  # no head-cut marker

    def test_dump_conversation_full_long_user_message(self):
        """Long user messages are dumped uncapped too (the live panel
        tail-caps them to 12 lines)."""
        tui, buf = make_tui()
        body = "\n".join(f"user line {i}" for i in range(30))
        tui.session.last_messages = [Message(role="user", content=body)]
        tui._dump_conversation()
        out = buf.getvalue()
        self.assertIn("user line 0", out)
        self.assertIn("user line 29", out)
        self.assertNotIn("…\n", out)

    def test_dump_conversation_still_filters_and_strips(self):
        """The full dump keeps the same display hygiene as the panel:
        injected prompts hidden, final-check blocks stripped."""
        from python_agent_harness import config

        tui, buf = make_tui()
        tui.session.last_messages = [
            Message(role="user", content="real question"),
            Message(role="user", content=config.NUDGE_MESSAGE),
            Message(
                role="assistant",
                content="Answer body.\n\n[FINAL CHECK]\n- Goal: x\n"
                "- Status: SUCCESS\n- Evidence: y",
            ),
        ]
        tui._dump_conversation()
        out = buf.getvalue()
        self.assertIn("real question", out)
        self.assertIn("Answer body.", out)
        self.assertNotIn(config.NUDGE_MESSAGE, out)
        self.assertNotIn("[FINAL CHECK]", out)

    def test_dump_separates_rounds_with_rule(self):
        """Each round after the first gets a rule separator in the
        scrollback dump, so rounds are visually distinct."""
        tui, buf = make_tui()
        tui.session.last_messages = [
            Message(role="user", content="first round question"),
            Message(role="assistant", content="first round answer"),
            Message(role="user", content="second round question"),
            Message(role="assistant", content="second round answer"),
            Message(role="user", content="third round question"),
            Message(role="assistant", content="third round answer"),
        ]
        tui._dump_conversation()
        out = buf.getvalue()
        self.assertIn("round 2", out)
        self.assertIn("round 3", out)
        self.assertIn("────", out)  # rule line
        # rounds still render in order, content untouched
        self.assertLess(out.index("first round question"), out.index("second round question"))
        self.assertLess(out.index("second round question"), out.index("third round question"))

    def test_dump_round_timestamp_when_recorded(self):
        """Live rounds carry their recorded start time on the separator."""
        import time as _time

        tui, buf = make_tui()
        tui.session.last_messages = [
            Message(role="user", content="first round"),
            Message(role="assistant", content="a1"),
            Message(role="user", content="second round"),
            Message(role="assistant", content="a2"),
        ]
        tui._round_times = [1_700_000_000.0, 1_700_000_100.0]
        tui._dump_conversation()
        out = buf.getvalue()
        expected = _time.strftime("%H:%M:%S", _time.localtime(1_700_000_100.0))
        self.assertIn(expected, out)
        # restored sessions have no times: separator without timestamp
        tui2, buf2 = make_tui()
        tui2.session.last_messages = [
            Message(role="user", content="first round"),
            Message(role="assistant", content="a1"),
            Message(role="user", content="second round"),
            Message(role="assistant", content="a2"),
        ]
        tui2._dump_conversation()
        out2 = buf2.getvalue()
        self.assertIn("round 2", out2)
        self.assertNotIn("·", out2)

    def test_dump_single_round_no_separator(self):
        """One round dumps without any round separator."""
        tui, buf = make_tui()
        tui.session.last_messages = [
            Message(role="user", content="only round"),
            Message(role="assistant", content="answer"),
        ]
        tui._dump_conversation()
        out = buf.getvalue()
        self.assertIn("only round", out)
        self.assertNotIn("round 2", out)

    def test_dump_reports_time_spent(self):
        """After the last response, the total time spent on the run is
        reported, mirroring the per-tool elapsed on tool results."""
        import time as _time

        tui, buf = make_tui()
        tui.session.last_messages = [
            Message(role="user", content="question"),
            Message(role="assistant", content="answer"),
        ]
        tui._run_start = _time.time() - 6.8
        tui._dump_conversation()
        out = buf.getvalue()
        self.assertIn("✓ time spent (6.8s):", out)

    def test_dump_no_time_spent_when_unrecorded(self):
        """No run start recorded (e.g. restored session) → no indicator."""
        tui, buf = make_tui()
        tui.session.last_messages = [
            Message(role="user", content="question"),
            Message(role="assistant", content="answer"),
        ]
        tui._dump_conversation()
        out = buf.getvalue()
        self.assertNotIn("time spent", out)

    def test_dump_shows_all_rounds_despite_round_start(self):
        """The end-of-run dump prints the FULL conversation even when
        round_start points past earlier rounds."""
        tui, buf = make_tui()
        tui.session.last_messages = [
            Message(role="user", content="first round question"),
            Message(role="assistant", content="first round answer"),
            Message(role="user", content="second round question"),
            Message(role="assistant", content="second round answer"),
        ]
        tui.round_start = 2
        tui._dump_conversation()
        out = buf.getvalue()
        self.assertIn("first round question", out)
        self.assertIn("first round answer", out)
        self.assertIn("second round question", out)
        self.assertIn("second round answer", out)

    # ------------------------------------------------------------------
    # conversation text / saved-body parsing
    # ------------------------------------------------------------------
    def test_conversation_text(self):
        """_conversation_text renders non-empty messages, skipping blanks."""
        tui, _ = make_tui()
        tui.session.last_messages = [
            Message(role="user", content="hello"),
            Message(role="assistant", content=""),
            Message(role="tool", content="result"),
        ]
        self.assertEqual(
            tui._conversation_text(),
            "**user**: hello\n\n**tool**: result",
        )

    def test_conversation_text_empty(self):
        tui, _ = make_tui()
        tui.session.last_messages = []
        self.assertEqual(tui._conversation_text(), "")

    def test_save_restore_round_trip_keeps_quoted_role_header(self):
        """A body line that looks like a block header must not split the
        message on restore: the renderer escapes it and the parser
        unescapes it, so quoting the save format (or pasting a
        transcript) round-trips unchanged."""
        tui, _ = make_tui()
        reply = "A block looks like:\n\n**user**: hello\n\nthat is the whole format."
        tui.session.last_messages = [
            Message(role="user", content="what does a saved session look like?"),
            Message(role="assistant", content=reply),
        ]
        body = tui._conversation_text()
        msgs = Tui._parse_saved_body(body)
        self.assertEqual([m.role for m in msgs], ["user", "assistant"])
        self.assertEqual(msgs[1].text(), reply)

    def test_save_restore_round_trip_keeps_literal_backslash(self):
        """An already-escaped-looking body line gains a second backslash
        on save, so the literal text survives the round trip exactly."""
        tui, _ = make_tui()
        reply = "escaped in the file as:\n\n\\**user**: hello"
        tui.session.last_messages = [Message(role="assistant", content=reply)]
        msgs = Tui._parse_saved_body(tui._conversation_text())
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].text(), reply)

    def test_parse_saved_body_drops_tool_blocks(self):
        """``**tool**:`` blocks are dropped from restored history,
        including a trailing tool block; all remaining messages must
        serialize to an API-valid payload."""
        body = (
            "**user**: find the files\n\n"
            "**assistant**: [tool calls: Glob, Read]\n\n"
            "**tool**: tests/test_agent.py\n"
            "tests/test_tui.py\n\n"
            "**assistant**: I found them.\n\n"
            "**tool**: trailing result"
        )
        msgs = Tui._parse_saved_body(body)
        self.assertEqual(
            [(m.role, m.text()) for m in msgs],
            [
                ("user", "find the files"),
                ("assistant", "[tool calls: Glob, Read]"),
                ("assistant", "I found them."),
            ],
        )
        for m in msgs:
            self.assertNotEqual(m.to_api()["role"], "tool")


if __name__ == "__main__":
    unittest.main()
