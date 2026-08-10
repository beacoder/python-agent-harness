"""Tests for filesystem tools: Edit (str + diff mode), Write diff capture."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

from python_agent_harness.tools.base import ToolContext
from python_agent_harness.tools.filesystem import (
    Edit,
    GlobTool,
    Grep,
    Insert,
    Mkdir,
    Read,
    Write,
    _apply_diff,
)


def _big_output(lines: int = 6000, width: int = 80) -> str:
    return "".join(f"line{i:04d}-{'x' * width}\n" for i in range(lines))


class FakeSession:
    """Minimal session double satisfying the ToolContext protocol."""

    def __init__(self) -> None:
        self.recorded_diffs: list[str] = []

    @property
    def project_dir(self) -> str:
        return "/tmp"

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

    def test_diff_with_no_newline_marker(self):
        """Diffs for files without a trailing newline use the
        '\\ No newline at end of file' marker (git-style); applying them
        must work, whether the marker line follows a bare line (as
        generated) or a newline-terminated one (as echoed by a model)."""
        from python_agent_harness.diffrender import unified_diff

        content = "a\nb"
        generated = unified_diff(content, "a\nc", "/x/x.txt")
        self.assertIn("\\ No newline at end of file", generated)
        self.assertEqual(_apply_diff(content, generated), "a\nc")

        git_style = (
            "--- a/x.txt\n"
            "+++ b/x.txt\n"
            "@@ -1,2 +1,2 @@\n"
            " a\n"
            "-b\n"
            "\\ No newline at end of file\n"
            "+c\n"
            "\\ No newline at end of file\n"
        )
        self.assertEqual(_apply_diff(content, git_style), "a\nc")

    def test_diff_in_fenced_code_block(self):
        """Models often wrap diffs in ```diff / ```patch fences (the
        gptel-agent Edit tool accepts these); the parser must skip the
        fence lines instead of failing on them."""
        content = "a\nb\nc\n"
        for fence in ("```diff", "```patch", "```"):
            diff = (
                f"{fence}\n"
                "--- a\n+++ b\n"
                "@@ -1,3 +1,3 @@\n"
                " a\n-b\n+B\n c\n"
                "```\n"
            )
            self.assertEqual(_apply_diff(content, diff), "a\nB\nc\n", fence)

    def test_glob_depth_zero_is_unlimited(self):
        """depth=0 must mean 'no limit' (like `tree -L 0`), never an
        empty result."""
        from python_agent_harness.tools.filesystem import _git_glob_results

        raw = "a.py\0sub/b.py\0sub/deep/c.py\0"
        out0 = _git_glob_results(raw, "/repo", "/repo", 0, "*.py")
        self.assertIn("sub/deep/c.py", out0)
        self.assertIn("a.py", out0)
        out1 = _git_glob_results(raw, "/repo", "/repo", 1, "*.py")
        self.assertIn("a.py", out1)
        self.assertNotIn("sub/b.py", out1)
        self.assertNotIn("sub/deep/c.py", out1)
        out_none = _git_glob_results(raw, "/repo", "/repo", None, "*.py")
        self.assertIn("sub/deep/c.py", out_none)

    def test_glob_depth_in_subdir_base(self):
        """depth is relative to the search base: entries outside the base
        subtree must never leak into the results."""
        from python_agent_harness.tools.filesystem import _git_glob_results

        raw = "a.py\0sub/b.py\0sub/deep/c.py\0sub/deep/deeper/d.py\0"
        out1 = _git_glob_results(raw, "/repo", "/repo/sub", 1, "*.py")
        self.assertIn("/repo/sub/b.py", out1)
        self.assertNotIn("/repo/a.py", out1)
        self.assertNotIn("/repo/sub/deep/c.py", out1)
        out2 = _git_glob_results(raw, "/repo", "/repo/sub", 2, "*.py")
        self.assertIn("/repo/sub/b.py", out2)
        self.assertIn("/repo/sub/deep/c.py", out2)
        self.assertNotIn("/repo/sub/deep/deeper/d.py", out2)


class TestSpool(unittest.TestCase):
    """Oversized Glob/Grep results must be spilled to a temp file (like
    `gptel-agent--truncate-buffer` in the elisp harness), with a preview
    plus a Read instruction, so no matches are lost."""

    def setUp(self):
        import python_agent_harness.tools.filesystem as fs

        self.fs = fs
        self.orig_dir = fs._spool_dir
        self.tmp = tempfile.TemporaryDirectory()
        fs._spool_dir = lambda: self.tmp.name

    def tearDown(self):
        self.fs._spool_dir = self.orig_dir
        self.tmp.cleanup()

    def test_small_output_passes_through(self):
        text = "small result\n"
        self.assertEqual(self.fs._spool(text, "grep"), text)

    def test_large_output_spills_to_temp_file(self):
        big = _big_output()
        self.assertGreater(len(big), self.fs.MAX_OUTPUT)
        result = self.fs._spool(big, "grep")
        self.assertIn("grep results too large", result)
        self.assertIn("Stored in:", result)
        self.assertIn(f"First {self.fs.SPOOL_LINES} lines", result)
        m = re.search(r'file_path="([^"]+)"', result)
        self.assertIsNotNone(m, result)
        self.assertTrue(os.path.isabs(m.group(1)))
        with open(m.group(1)) as f:
            self.assertEqual(f.read(), big)
        self.assertIn("line0000-", result)
        self.assertNotIn("line5999-", result)

    def test_large_glob_output_spills(self):
        big = "\n".join(f"/repo/f{i:05d}.py" for i in range(20000))
        self.assertGreater(len(big), self.fs.MAX_OUTPUT)
        result = self.fs._git_glob_results(big, "/repo", "/repo", None, "*.py")
        self.assertIn("glob results too large", result)
        m = re.search(r'file_path="([^"]+)"', result)
        self.assertIsNotNone(m, result)
        with open(m.group(1)) as f:
            self.assertEqual(f.read(), big + "\n")

    def test_grep_tool_spills_large_result(self):
        import shutil

        if not shutil.which("grep"):
            self.skipTest("grep not available")
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "data.txt"), "w") as f:
                # 1500 matching lines; --max-count=1000 + -C15 context
                # inflates output well beyond MAX_OUTPUT
                f.write(
                    "".join(
                        f"line{i:04d}-needle-{'x' * 300}\n" for i in range(1500)
                    )
                )
            ctx, _ = make_ctx()
            result = Grep().run(
                {"regex": "needle", "path": d, "context_lines": 15}, ctx
            )
            self.assertIn("grep results too large", result)
            m = re.search(r'file_path="([^"]+)"', result)
            self.assertIsNotNone(m, result)
            self.assertGreater(os.path.getsize(m.group(1)), self.fs.MAX_OUTPUT)

    def test_spool_falls_back_to_truncation_when_tempdir_unwritable(self):
        self.fs._spool_dir = lambda: "/dev/null/definitely-not-a-dir"
        big = _big_output()
        result = self.fs._spool(big, "grep")
        self.assertIn("[truncated grep]", result)
        self.assertNotIn("Stored in:", result)

    def test_grep_out_formats_backend_errors(self):
        """A failing backend (exit code >= 2) must surface as an explicit
        error with the backend's stderr, not as an empty result."""
        import subprocess

        proc = subprocess.CompletedProcess([], returncode=2, stdout="boom\n")
        out = self.fs._grep_out(proc, "rg")
        self.assertIn("Error: search failed with exit-code 2", out)
        self.assertIn("boom", out)

    def test_grep_context_lines_clamped_to_15(self):
        """context_lines beyond 15 is clamped (schema maximum), and a
        huge context value must not crash the search."""
        import shutil

        if not shutil.which("grep") and not shutil.which("rg"):
            self.skipTest("no grep backend available")
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "data.txt"), "w") as f:
                f.write("one\ntwo\nthree\nneedle\nfour\nfive\nsix\n")
            ctx, _ = make_ctx()
            result = Grep().run(
                {"regex": "needle", "path": d, "context_lines": 99}, ctx
            )
            self.assertIn("needle", result)
            self.assertNotIn("Error", result)


class TestReadTool(unittest.TestCase):
    """Read mirrors `gptel-agent--read-file-lines`: whole-file reads are
    refused above READ_SIZE_LIMIT, line ranges are streamed instead of
    loading the file, and oversized range results spill to a temp file
    rather than being silently truncated."""

    def setUp(self):
        import python_agent_harness.tools.filesystem as fs

        self.fs = fs
        self.orig_dir = fs._spool_dir
        self.tmp = tempfile.TemporaryDirectory()
        fs._spool_dir = lambda: self.tmp.name

    def tearDown(self):
        self.fs._spool_dir = self.orig_dir
        self.tmp.cleanup()

    def _write(self, lines: int, width: int = 20) -> str:
        p = os.path.join(self.tmp.name, f"f{lines}x{width}.txt")
        with open(p, "w") as f:
            for i in range(lines):
                f.write(f"line{i:05d}-{'x' * width}\n")
        return p

    def test_full_read_small_file_returns_raw_content(self):
        p = self._write(5)
        out = Read().run({"file_path": p}, ToolContext())
        self.assertIn("line00000-", out)
        self.assertIn("line00004-", out)
        self.assertNotIn("Showing lines", out)

    def test_full_read_too_large_refuses_like_gptel(self):
        p = self._write(20000)
        self.assertGreater(os.path.getsize(p), self.fs.READ_SIZE_LIMIT)
        out = Read().run({"file_path": p}, ToolContext())
        self.assertIn("File is too large", out)
        self.assertIn("specify a line range", out)
        self.assertNotIn("line00000-", out)

    def test_range_read_streams_selected_lines(self):
        p = self._write(1000)
        out = Read().run(
            {"file_path": p, "start_line": 5, "end_line": 7}, ToolContext()
        )
        self.assertIn("Showing lines 5-7:", out)
        self.assertNotIn("of 1000", out)   # total unknown when EOF not reached
        self.assertIn("line00004-", out)
        self.assertIn("line00006-", out)
        self.assertNotIn("line00007-", out)

    def test_end_line_only_starts_from_line_1(self):
        p = self._write(10)
        out = Read().run({"file_path": p, "end_line": 3}, ToolContext())
        self.assertIn("Showing lines 1-3:", out)
        self.assertIn("line00000-", out)
        self.assertIn("line00002-", out)
        self.assertNotIn("line00003-", out)

    def test_start_line_only_reads_to_eof_with_total(self):
        p = self._write(10)
        out = Read().run({"file_path": p, "start_line": 8}, ToolContext())
        self.assertIn("Showing lines 8-10 of 10:", out)
        self.assertIn("line00007-", out)
        self.assertIn("line00009-", out)

    def test_range_read_on_file_above_limit_is_allowed(self):
        """The 400 KB gate applies only to whole-file reads, not ranges."""
        p = self._write(20000, width=80)   # > 1 MB
        self.assertGreater(os.path.getsize(p), self.fs.READ_SIZE_LIMIT)
        out = Read().run(
            {"file_path": p, "start_line": 1, "end_line": 3}, ToolContext()
        )
        self.assertNotIn("Error", out)
        self.assertIn("Showing lines 1-3:", out)
        self.assertIn("line00000-", out)

    def test_full_read_at_exact_limit_is_allowed(self):
        p = os.path.join(self.tmp.name, "exact.txt")
        with open(p, "w") as f:
            f.write("x" * self.fs.READ_SIZE_LIMIT)
        out = Read().run({"file_path": p}, ToolContext())
        self.assertEqual(len(out), self.fs.READ_SIZE_LIMIT)
        self.assertNotIn("Error", out)

    def test_no_trailing_newline_counts_lines_correctly(self):
        p = os.path.join(self.tmp.name, "nonl.txt")
        with open(p, "w") as f:
            f.write("a\nb\nc")
        out = Read().run({"file_path": p, "start_line": 2, "end_line": 3}, ToolContext())
        self.assertIn("Showing lines 2-3 of 3:", out)
        self.assertIn("b\nc", out)

    def test_negative_start_line_clamped_to_1(self):
        p = self._write(10)
        out = Read().run(
            {"file_path": p, "start_line": -5, "end_line": 2}, ToolContext()
        )
        self.assertIn("Showing lines 1-2:", out)
        self.assertIn("line00000-", out)

    def test_missing_file_errors(self):
        out = Read().run(
            {"file_path": os.path.join(self.tmp.name, "nope.txt")}, ToolContext()
        )
        self.assertIn("Error", out)

    def test_empty_file_full_read_returns_empty_string(self):
        p = os.path.join(self.tmp.name, "empty.txt")
        open(p, "w").close()
        self.assertEqual(Read().run({"file_path": p}, ToolContext()), "")

    def test_empty_file_range_read_errors_without_none_in_message(self):
        p = os.path.join(self.tmp.name, "empty.txt")
        open(p, "w").close()
        out = Read().run({"file_path": p, "start_line": 1, "end_line": 5}, ToolContext())
        self.assertIn("Error", out)
        self.assertNotIn("None", out)
        out = Read().run({"file_path": p, "start_line": 1}, ToolContext())
        self.assertIn("Error", out)
        self.assertNotIn("None", out)

    def test_invalid_utf8_bytes_replaced_not_fatal(self):
        p = os.path.join(self.tmp.name, "latin.txt")
        with open(p, "wb") as f:
            f.write(b"caf\xe9 \xff line\n")
        out = Read().run({"file_path": p}, ToolContext())
        self.assertIn("line", out)
        self.assertNotIn("Error", out)

    def test_symlink_resolved_before_size_check(self):
        target = os.path.join(self.tmp.name, "target.txt")
        link = os.path.join(self.tmp.name, "link.txt")
        with open(target, "w") as f:
            f.write("through link\n")
        os.symlink(target, link)
        out = Read().run({"file_path": link}, ToolContext())
        self.assertIn("through link", out)

    def test_read_spool_falls_back_to_truncation_when_tempdir_unwritable(self):
        self.fs._spool_dir = lambda: "/dev/null/definitely-not-a-dir"
        p = self._write(20000, width=80)
        out = Read().run(
            {"file_path": p, "start_line": 1, "end_line": 20000}, ToolContext()
        )
        self.assertIn("[truncated read]", out)
        self.assertNotIn("Stored in:", out)

    def test_range_read_reports_total_when_eof_reached(self):
        p = self._write(10)
        out = Read().run(
            {"file_path": p, "start_line": 8, "end_line": 999}, ToolContext()
        )
        self.assertIn("Showing lines 8-10 of 10:", out)
        self.assertIn("line00009-", out)

    def test_range_read_errors_on_start_gt_end(self):
        p = self._write(10)
        out = Read().run(
            {"file_path": p, "start_line": 5, "end_line": 3}, ToolContext()
        )
        self.assertIn("start_line 5 > end_line 3", out)

    def test_range_read_errors_when_start_beyond_eof(self):
        p = self._write(10)
        out = Read().run({"file_path": p, "start_line": 99}, ToolContext())
        self.assertIn("start_line 99 > end_line 10", out)

    def test_large_range_spills_to_temp_file(self):
        p = self._write(20000, width=80)
        out = Read().run(
            {"file_path": p, "start_line": 1, "end_line": 20000}, ToolContext()
        )
        self.assertIn("read results too large", out)
        m = re.search(r'file_path="([^"]+)"', out)
        self.assertIsNotNone(m, out)
        with open(m.group(1)) as f:
            content = f.read()
        self.assertIn("Showing lines 1-20000 of 20000:", content)
        self.assertIn("line19999-", content)

    def test_read_directory_errors(self):
        out = Read().run({"file_path": self.tmp.name}, ToolContext())
        self.assertIn("Error", out)

    @unittest.skipIf(
        hasattr(os, "geteuid") and os.geteuid() == 0,
        "root bypasses file permission checks",
    )
    def test_unreadable_file_returns_error_not_crash(self):
        p = os.path.join(self.tmp.name, "secret.txt")
        with open(p, "w") as f:
            f.write("secret\n")
        os.chmod(p, 0o000)
        try:
            out = Read().run({"file_path": p}, ToolContext())
        finally:
            os.chmod(p, 0o644)
        self.assertIn("Error", out)
        self.assertNotIn("Traceback", out)


class TestGlobGrepTools(unittest.TestCase):
    """End-to-end Glob/Grep behavior against real backends: the git-aware
    branch (git ls-files / git grep) and the tree / plain-grep fallbacks,
    plus error paths.  Backend-dependent tests skip when the executable
    is missing."""

    def setUp(self):
        self.ctx = ToolContext()
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _mkdir(self, *parts) -> str:
        p = os.path.join(self.tmp.name, *parts)
        os.makedirs(p, exist_ok=True)
        return p

    @unittest.skipUnless(shutil.which("tree"), "tree not available")
    def test_glob_tree_fallback_lists_files_and_depth(self):
        d = self._mkdir("proj")
        open(os.path.join(d, "a.py"), "w").close()
        self._mkdir("proj", "sub")
        open(os.path.join(d, "sub", "b.py"), "w").close()
        out = GlobTool().run({"pattern": "*.py", "path": d}, self.ctx)
        self.assertIn(os.path.join(d, "a.py"), out)
        self.assertIn(os.path.join(d, "sub", "b.py"), out)
        out1 = GlobTool().run({"pattern": "*.py", "path": d, "depth": 1}, self.ctx)
        self.assertIn(os.path.join(d, "a.py"), out1)
        self.assertNotIn(os.path.join(d, "sub", "b.py"), out1)

    @unittest.skipUnless(shutil.which("git"), "git not available")
    def test_glob_and_grep_use_git_backend_in_repo(self):
        repo = self._mkdir("repo")
        subprocess.run(["git", "init", "-q", repo], check=True)
        for name, content in (("a.py", "hello world\n"), ("b.txt", "nope\n")):
            with open(os.path.join(repo, name), "w") as f:
                f.write(content)
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        out = GlobTool().run({"pattern": "*", "path": repo}, self.ctx)
        self.assertIn(os.path.join(repo, "a.py"), out)
        self.assertIn(os.path.join(repo, "b.txt"), out)
        out = Grep().run({"regex": "hello", "path": repo}, self.ctx)
        self.assertIn("a.py", out)
        self.assertIn("hello", out)
        self.assertNotIn("b.txt", out)
        out = Grep().run(
            {"regex": "hello", "path": repo, "context_lines": 2}, self.ctx
        )
        self.assertIn("a.py", out)
        self.assertIn("hello", out)

    def test_glob_nonexistent_path_errors(self):
        out = GlobTool().run(
            {"pattern": "*", "path": os.path.join(self.tmp.name, "nope")}, self.ctx
        )
        self.assertIn("Error", out)

    def test_glob_empty_pattern_errors(self):
        out = GlobTool().run({"pattern": "", "path": self.tmp.name}, self.ctx)
        self.assertIn("Error", out)

    def test_grep_single_file_path(self):
        d = self._mkdir("proj")
        p = os.path.join(d, "a.py")
        with open(p, "w") as f:
            f.write("alpha\nbeta\nalpha\n")
        out = Grep().run({"regex": "alpha", "path": p}, self.ctx)
        self.assertIn("alpha", out)
        self.assertIn("1", out)

    def test_grep_glob_filter_restricts_files(self):
        d = self._mkdir("proj")
        with open(os.path.join(d, "a.py"), "w") as f:
            f.write("needle\n")
        with open(os.path.join(d, "a.md"), "w") as f:
            f.write("needle\n")
        out = Grep().run({"regex": "needle", "path": d, "glob": "*.py"}, self.ctx)
        self.assertIn("a.py", out)
        self.assertNotIn("a.md", out)

    def test_grep_nonexistent_path_errors(self):
        out = Grep().run(
            {"regex": "x", "path": os.path.join(self.tmp.name, "nope")}, self.ctx
        )
        self.assertIn("Error", out)

    @mock.patch("shutil.which", return_value=None)
    def test_glob_errors_when_tree_missing(self, _which):
        d = self._mkdir("proj")
        out = GlobTool().run({"pattern": "*.py", "path": d}, self.ctx)
        self.assertIn("Executable `tree` not found", out)

    @mock.patch("shutil.which", return_value=None)
    def test_grep_errors_when_no_backend_available(self, _which):
        out = Grep().run({"regex": "x", "path": self.tmp.name}, self.ctx)
        self.assertIn("ripgrep/grep/git-grep not available", out)


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


class TestInsertTool(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "f.txt")
        with open(self.path, "w") as f:
            f.write("a\nb\nc\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_insert_after_line_number(self):
        Insert().run({"path": self.path, "line_number": 2, "new_str": "X"}, ToolContext())
        self.assertEqual(open(self.path).read(), "a\nb\nX\nc\n")

    def test_insert_at_beginning(self):
        Insert().run({"path": self.path, "line_number": 0, "new_str": "Z"}, ToolContext())
        self.assertEqual(open(self.path).read(), "Z\na\nb\nc\n")

    def test_insert_at_end(self):
        Insert().run({"path": self.path, "line_number": -1, "new_str": "Y"}, ToolContext())
        self.assertEqual(open(self.path).read(), "a\nb\nc\nY\n")

    def test_insert_adds_missing_trailing_newline(self):
        Insert().run({"path": self.path, "line_number": 1, "new_str": "X"}, ToolContext())
        self.assertEqual(open(self.path).read(), "a\nX\nb\nc\n")


class TestMkdirTool(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_create_directory(self):
        out = Mkdir().run({"parent": self.tmp.name, "name": "sub"}, ToolContext())
        self.assertIn("created/verified", out)
        self.assertTrue(os.path.isdir(os.path.join(self.tmp.name, "sub")))

    def test_create_nested_directory(self):
        out = Mkdir().run(
            {"parent": self.tmp.name, "name": "sub/deep"}, ToolContext()
        )
        self.assertIn("created/verified", out)
        self.assertTrue(os.path.isdir(os.path.join(self.tmp.name, "sub/deep")))

    def test_existing_directory_is_noop(self):
        d = os.path.join(self.tmp.name, "sub")
        os.makedirs(d)
        out = Mkdir().run({"parent": self.tmp.name, "name": "sub"}, ToolContext())
        self.assertIn("created/verified", out)


if __name__ == "__main__":
    unittest.main()
