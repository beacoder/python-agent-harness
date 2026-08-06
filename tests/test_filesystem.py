"""Tests for filesystem tools: Edit (str + diff mode), Write diff capture."""

from __future__ import annotations

import os
import tempfile
import unittest

from python_agent_harness.tools.base import ToolContext
from python_agent_harness.tools.filesystem import Edit, Write, _apply_diff


class FakeSession:
    """Minimal session double satisfying the ToolContext protocol."""

    def __init__(self) -> None:
        self.snapshots: list[tuple[str, str]] = []
        self.invalidated: list[str] = []
        self.recorded_diffs: list[str] = []

    @property
    def project_dir(self) -> str:
        return "/tmp"

    def guard_path(self, path: str, tool_name: str) -> None:
        pass

    def snapshot(self, path: str, tool: str) -> None:
        self.snapshots.append((path, tool))

    def record_absent(self, path: str, tool: str) -> None:
        pass

    def invalidate_cache(self, path: str) -> None:
        self.invalidated.append(path)

    def record_diff(self, diff_text: str) -> None:
        self.recorded_diffs.append(diff_text)


def make_ctx() -> tuple[ToolContext, FakeSession]:
    sess = FakeSession()
    return ToolContext(sess), sess


class TestApplyDiff(unittest.TestCase):
    def test_replace_line(self):
        content = "line1\nline2\nline3\n"
        diff = (
            "--- a\n+++ b\n@@ -1,3 +1,3 @@\n"
            " line1\n-line2\n+lineTWO\n line3\n"
        )
        self.assertEqual(_apply_diff(content, diff), "line1\nlineTWO\nline3\n")

    def test_add_only(self):
        content = "a\nb\nc\n"
        diff = "@@ -1,2 +1,3 @@\n a\n+NEW\n b\n"
        self.assertEqual(_apply_diff(content, diff), "a\nNEW\nb\nc\n")

    def test_remove_only(self):
        content = "a\nb\nc\n"
        diff = "@@ -1,2 +1,1 @@\n a\n-b\n"
        self.assertEqual(_apply_diff(content, diff), "a\nc\n")

    def test_multiple_hunks(self):
        content = "a\nb\nc\nd\ne\n"
        diff = "@@ -1,1 +1,1 @@\n-a\n+A\n@@ -4,1 +4,1 @@\n-d\n+D\n"
        self.assertEqual(_apply_diff(content, diff), "A\nb\nc\nD\ne\n")

    def test_context_mismatch_raises(self):
        content = "a\nb\nc\n"
        diff = "@@ -1,2 +1,2 @@\n a\n-ZZZ\n+B\n"
        with self.assertRaises(ValueError):
            _apply_diff(content, diff)

    def test_no_hunks_raises(self):
        with self.assertRaises(ValueError):
            _apply_diff("a\n", "not a diff")


class TestEditTool(unittest.TestCase):
    def test_old_str_replace(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "f.txt")
            with open(path, "w") as f:
                f.write("hello world\n")
            ctx, sess = make_ctx()
            result = Edit().run(
                {"path": path, "old_str": "hello", "new_str": "goodbye"}, ctx
            )
            self.assertIn("Successfully replaced", result)
            with open(path) as f:
                self.assertEqual(f.read(), "goodbye world\n")
            self.assertEqual(len(sess.recorded_diffs), 1)
            self.assertIn("-hello world", sess.recorded_diffs[0])
            self.assertIn("+goodbye world", sess.recorded_diffs[0])

    def test_diff_mode_applies_correctly(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "f.txt")
            with open(path, "w") as f:
                f.write("line1\nline2\nline3\n")
            ctx, sess = make_ctx()
            diff = (
                "--- a/f.txt\n+++ b/f.txt\n@@ -1,3 +1,3 @@\n"
                " line1\n-line2\n+lineTWO\n line3\n"
            )
            result = Edit().run(
                {"path": path, "new_str": diff, "diff": True}, ctx
            )
            self.assertIn("Successfully replaced", result)
            with open(path) as f:
                self.assertEqual(f.read(), "line1\nlineTWO\nline3\n")
            self.assertEqual(len(sess.recorded_diffs), 1)

    def test_diff_mode_mismatch_reports_error_and_does_not_write(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "f.txt")
            original = "line1\nline2\nline3\n"
            with open(path, "w") as f:
                f.write(original)
            ctx, sess = make_ctx()
            bad_diff = "@@ -1,2 +1,2 @@\n line1\n-NOPE\n+lineTWO\n"
            result = Edit().run(
                {"path": path, "new_str": bad_diff, "diff": True}, ctx
            )
            self.assertTrue(result.startswith("Error:"))
            with open(path) as f:
                self.assertEqual(f.read(), original)  # unchanged on failure
            self.assertEqual(sess.recorded_diffs, [])  # no diff recorded on failure

    def test_no_diff_recorded_when_content_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "f.txt")
            with open(path, "w") as f:
                f.write("same\n")
            ctx, sess = make_ctx()
            Edit().run({"path": path, "old_str": "same", "new_str": "same"}, ctx)
            self.assertEqual(sess.recorded_diffs, [])

    def test_old_str_not_found(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "f.txt")
            with open(path, "w") as f:
                f.write("hello\n")
            ctx, _ = make_ctx()
            result = Edit().run(
                {"path": path, "old_str": "missing", "new_str": "x"}, ctx
            )
            self.assertIn("not found", result)

    def test_old_str_not_unique(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "f.txt")
            with open(path, "w") as f:
                f.write("dup\ndup\n")
            ctx, _ = make_ctx()
            result = Edit().run(
                {"path": path, "old_str": "dup", "new_str": "x"}, ctx
            )
            self.assertIn("not unique", result)


class TestWriteTool(unittest.TestCase):
    def test_new_file_shows_all_lines_added(self):
        with tempfile.TemporaryDirectory() as d:
            ctx, sess = make_ctx()
            result = Write().run(
                {"path": d, "filename": "new.txt", "content": "hi\n"}, ctx
            )
            self.assertIn("Created file", result)
            # a brand-new file is shown as an all-added diff
            self.assertEqual(len(sess.recorded_diffs), 1)
            self.assertIn("+hi", sess.recorded_diffs[0])

    def test_overwrite_existing_file_records_diff(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "f.txt")
            with open(path, "w") as f:
                f.write("old content\n")
            ctx, sess = make_ctx()
            Write().run(
                {"path": d, "filename": "f.txt", "content": "new content\n"}, ctx
            )
            self.assertEqual(len(sess.recorded_diffs), 1)
            self.assertIn("-old content", sess.recorded_diffs[0])
            self.assertIn("+new content", sess.recorded_diffs[0])

    def test_overwrite_with_identical_content_no_diff(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "f.txt")
            with open(path, "w") as f:
                f.write("same\n")
            ctx, sess = make_ctx()
            Write().run({"path": d, "filename": "f.txt", "content": "same\n"}, ctx)
            self.assertEqual(sess.recorded_diffs, [])


if __name__ == "__main__":
    unittest.main()
