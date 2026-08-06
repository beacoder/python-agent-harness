"""Tests for the unified-diff generation/rendering helper."""

from __future__ import annotations

import io
import unittest

from rich.console import Console

from python_agent_harness.diffrender import render_diff, unified_diff


class TestUnifiedDiff(unittest.TestCase):
    def test_identical_content_returns_empty(self):
        self.assertEqual(unified_diff("same\n", "same\n", "f.py"), "")

    def test_changed_content_returns_diff(self):
        diff = unified_diff("a\nb\nc\n", "a\nB\nc\n", "f.py")
        self.assertIn("-b", diff)
        self.assertIn("+B", diff)
        self.assertIn("f.py", diff)

    def test_new_file_from_empty(self):
        diff = unified_diff("", "new content\n", "f.py")
        self.assertIn("+new content", diff)


class TestRenderDiff(unittest.TestCase):
    def _render_to_text(self, diff_text: str) -> str:
        buf = io.StringIO()
        console = Console(file=buf, width=100, force_terminal=False)
        console.print(render_diff(diff_text))
        return buf.getvalue()

    def test_render_shows_added_and_removed_lines(self):
        diff = unified_diff("old\n", "new\n", "f.py")
        out = self._render_to_text(diff)
        self.assertIn("-old", out)
        self.assertIn("+new", out)

    def test_render_empty_diff_shows_placeholder(self):
        out = self._render_to_text("")
        self.assertIn("no changes", out)

    def test_render_truncates_long_diff(self):
        lines = [f"+line{i}" for i in range(500)]
        diff = "@@ -1,1 +1,500 @@\n" + "\n".join(lines)
        out = self._render_to_text(diff)
        self.assertIn("truncated", out)


if __name__ == "__main__":
    unittest.main()
