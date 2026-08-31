"""Tests for filesystem tools: Edit (str + diff mode), Write diff capture."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from python_agent_harness.tools.base import ToolContext
from python_agent_harness.tools.edit_mac import EditMac
from python_agent_harness.tools.filesystem import (
    Edit,
    GlobTool,
    Grep,
    Insert,
    Mkdir,
    Read,
    Write,
    _fix_patch_headers,
    _strip_diff_fence,
)


def edit_tool() -> Edit:
    """The Edit tool active on this platform: Linux uses the patch
    binary, macOS the built-in Python diff applier (Apple's BSD patch
    rejects well-formed hunks that GNU patch accepts)."""
    return EditMac() if sys.platform == "darwin" else Edit()


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


class TestPatchHelpers(unittest.TestCase):
    """Diff/patch-mode helpers, mirroring `gptel-agent--edit-files`:
    ```diff fence removal and hunk-header line-count fixing."""

    def test_fix_patch_headers_recomputes_counts(self):
        # Header claims -9,9 +9,9 but the body has 2 context, 1 removed,
        # 1 added -> orig=3, new=3.
        diff = "@@ -1,9 +1,9 @@\n a\n-b\n+B\n c\n"
        fixed = _fix_patch_headers(diff)
        self.assertIn("@@ -1,3 +1,3 @@", fixed)
        # Body is preserved verbatim.
        self.assertIn(" a\n-b\n+B\n c\n", fixed)

    def test_fix_patch_headers_multiple_hunks(self):
        # Each hunk's counts are recomputed independently; start lines kept.
        diff = "@@ -1,9 +1,9 @@\n-a\n+A\n@@ -5,9 +5,9 @@\n-e\n+E\n"
        fixed = _fix_patch_headers(diff)
        self.assertIn("@@ -1,1 +1,1 @@", fixed)
        self.assertIn("@@ -5,1 +5,1 @@", fixed)

    def test_fix_patch_headers_ignores_file_header_lines(self):
        diff = "--- a/x\n+++ b/x\n@@ -1,1 +1,2 @@\n a\n+NEW\n"
        fixed = _fix_patch_headers(diff)
        # +++/--- lines must not be miscounted as add/remove lines.
        self.assertIn("--- a/x\n", fixed)
        self.assertIn("+++ b/x\n", fixed)
        self.assertIn("@@ -1,1 +1,2 @@", fixed)

    def test_fix_patch_headers_counts_dashed_content_lines(self):
        # Removed/added lines whose content starts with --/++ (rendered
        # ---/+++) are hunk body, not file headers: they must be counted.
        diff = "@@ -1,9 +1,9 @@\n a\n---removed--line\n+++added++line\n c\n"
        fixed = _fix_patch_headers(diff)
        self.assertIn("@@ -1,3 +1,3 @@", fixed)
        self.assertIn("---removed--line\n", fixed)
        self.assertIn("+++added++line\n", fixed)

    def test_fix_patch_headers_counts_dashed_line_ending_a_hunk(self):
        # A dashed content line as the hunk's LAST body line is followed
        # directly by the next hunk header, which looks exactly like a
        # file header pair to a plain "peek for @@" check: it must still
        # be counted, or the (correct) input header is rewritten wrong
        # and `patch` rejects the whole diff.
        diff = (
            "@@ -2,3 +2,2 @@\n   id INT\n );\n--- old comment\n@@ -8,1 +8,1 @@\n--- tail\n+-- new\n"
        )
        fixed = _fix_patch_headers(diff)
        self.assertIn("@@ -2,3 +2,2 @@", fixed)
        self.assertIn("--- old comment\n", fixed)

    def test_fix_patch_headers_counts_dashed_pair_ending_a_hunk(self):
        # Same, for a ---/+++ pair whose content carries no space after
        # the marker (real file headers always name a path).
        diff = "@@ -1,3 +1,3 @@\n a\n---removed\n+++added\n@@ -20,1 +20,1 @@\n z\n"
        fixed = _fix_patch_headers(diff)
        self.assertIn("@@ -1,2 +1,2 @@", fixed)
        self.assertIn("---removed\n", fixed)
        self.assertIn("+++added\n", fixed)

    def test_fix_patch_headers_multifile_headers_not_counted(self):
        # A new file's ---/+++ header pair (followed by a hunk header) ends
        # the previous hunk's body instead of being miscounted as content.
        diff = "@@ -1,9 +1,9 @@\n a\n-b\n+B\n c\n--- a/f2\n+++ b/f2\n@@ -1,2 +1,2 @@\n x\n y\n"
        fixed = _fix_patch_headers(diff)
        self.assertIn("@@ -1,3 +1,3 @@", fixed)
        self.assertIn("--- a/f2\n", fixed)
        self.assertIn("+++ b/f2\n", fixed)
        self.assertIn("@@ -1,2 +1,2 @@", fixed)

    def test_fix_patch_headers_passes_through_headerless_counts(self):
        # A header without explicit counts is left untouched.
        diff = "@@ -1 +1 @@\n a\n"
        self.assertEqual(_fix_patch_headers(diff), diff)

    def test_strip_diff_fence_removes_diff_fence(self):
        text = "```diff\n--- a\n+++ b\n@@ -1,1 +1,1 @@\n-a\n+b\n```\n"
        stripped = _strip_diff_fence(text)
        self.assertFalse(stripped.startswith("```"))
        self.assertNotIn("```", stripped)
        self.assertIn("@@ -1,1 +1,1 @@", stripped)

    def test_strip_diff_fence_leaves_plain_and_other_fences(self):
        plain = "--- a\n+++ b\n@@ -1,1 +1,1 @@\n-a\n+b\n"
        self.assertEqual(_strip_diff_fence(plain), plain)
        # Only ```diff is stripped (mirrors the elisp); ```patch is left
        # for `patch` to handle/reject.
        patch_fence = "```patch\n--- a\n+++ b\n```\n"
        self.assertEqual(_strip_diff_fence(patch_fence), patch_fence)


class TestGitGlobResults(unittest.TestCase):
    """Depth filtering of `git ls-files` output, mirroring the git branch
    of `gptel-agent-harness-tools--glob` (Emacs `natnump' semantics)."""

    def test_depth_none_is_unlimited(self):
        from python_agent_harness.tools.filesystem import _git_glob_results

        raw = "a.py\0sub/b.py\0sub/deep/c.py\0"
        out = _git_glob_results(raw, "/repo", "/repo", None)
        self.assertIn("/repo/a.py", out)
        self.assertIn("/repo/sub/b.py", out)
        self.assertIn("/repo/sub/deep/c.py", out)

    def test_natnump_semantics(self):
        # Emacs `natnump' = non-negative integer.  Critically, bool must
        # NOT count (isinstance(True, int) is True in Python) or depth=True
        # would silently act like depth=1.
        from python_agent_harness.tools.filesystem import _natnump

        self.assertTrue(_natnump(0))
        self.assertTrue(_natnump(3))
        self.assertFalse(_natnump(-1))
        self.assertFalse(_natnump(None))
        self.assertFalse(_natnump(True))
        self.assertFalse(_natnump(1.0))

    def test_depth_one_keeps_only_top_level(self):
        from python_agent_harness.tools.filesystem import _git_glob_results

        raw = "a.py\0sub/b.py\0sub/deep/c.py\0"
        out = _git_glob_results(raw, "/repo", "/repo", 1)
        self.assertIn("/repo/a.py", out)
        self.assertNotIn("/repo/sub/b.py", out)
        self.assertNotIn("/repo/sub/deep/c.py", out)

    def test_depth_relative_to_search_base(self):
        from python_agent_harness.tools.filesystem import _git_glob_results

        # git ls-files with pathspec "sub/*" only returns entries under sub/.
        raw = "sub/b.py\0sub/deep/c.py\0sub/deep/deeper/d.py\0"
        out1 = _git_glob_results(raw, "/repo", "/repo/sub", 1)
        self.assertIn("/repo/sub/b.py", out1)
        self.assertNotIn("/repo/sub/deep/c.py", out1)
        out2 = _git_glob_results(raw, "/repo", "/repo/sub", 2)
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
        self.fs._spooled_files.clear()
        self.tmp.cleanup()

    def test_small_output_passes_through(self):
        text = "small result\n"
        self.assertEqual(self.fs._spool(text, "grep"), text)

    def test_cleanup_spooled_files(self):
        big = _big_output()
        result = self.fs._spool(big, "grep")
        m = re.search(r'file_path="([^"]+)"', result)
        self.assertIsNotNone(m, result)
        self.assertTrue(os.path.exists(m.group(1)))
        self.fs.cleanup_spooled_files()
        self.assertFalse(os.path.exists(m.group(1)))
        self.assertEqual(self.fs._spooled_files, [])

    def test_cleanup_skips_missing_files(self):
        big = _big_output()
        result = self.fs._spool(big, "grep")
        m = re.search(r'file_path="([^"]+)"', result)
        os.remove(m.group(1))
        self.fs.cleanup_spooled_files()  # must not raise
        self.assertEqual(self.fs._spooled_files, [])

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
        result = self.fs._git_glob_results(big, "/repo", "/repo", None)
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
                f.write("".join(f"line{i:04d}-needle-{'x' * 300}\n" for i in range(1500)))
            ctx, _ = make_ctx()
            result = Grep().run({"regex": "needle", "path": d, "context_lines": 15}, ctx)
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
            result = Grep().run({"regex": "needle", "path": d, "context_lines": 99}, ctx)
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
        out = Read().run({"file_path": p, "start_line": 5, "end_line": 7}, ToolContext())
        self.assertIn("Showing lines 5-7:", out)
        self.assertNotIn("of 1000", out)  # total unknown when EOF not reached
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
        p = self._write(20000, width=80)  # > 1 MB
        self.assertGreater(os.path.getsize(p), self.fs.READ_SIZE_LIMIT)
        out = Read().run({"file_path": p, "start_line": 1, "end_line": 3}, ToolContext())
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
        out = Read().run({"file_path": p, "start_line": -5, "end_line": 2}, ToolContext())
        self.assertIn("Showing lines 1-2:", out)
        self.assertIn("line00000-", out)

    def test_missing_file_errors(self):
        out = Read().run({"file_path": os.path.join(self.tmp.name, "nope.txt")}, ToolContext())
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
        out = Read().run({"file_path": p, "start_line": 1, "end_line": 20000}, ToolContext())
        self.assertIn("[truncated read]", out)
        self.assertNotIn("Stored in:", out)

    def test_range_read_reports_total_when_eof_reached(self):
        p = self._write(10)
        out = Read().run({"file_path": p, "start_line": 8, "end_line": 999}, ToolContext())
        self.assertIn("Showing lines 8-10 of 10:", out)
        self.assertIn("line00009-", out)

    def test_range_read_errors_on_start_gt_end(self):
        p = self._write(10)
        out = Read().run({"file_path": p, "start_line": 5, "end_line": 3}, ToolContext())
        self.assertIn("start_line 5 > end_line 3", out)

    def test_range_read_errors_when_start_beyond_eof(self):
        p = self._write(10)
        out = Read().run({"file_path": p, "start_line": 99}, ToolContext())
        self.assertIn("start_line 99 > end_line 10", out)

    def test_large_range_spills_to_temp_file(self):
        p = self._write(20000, width=80)
        out = Read().run({"file_path": p, "start_line": 1, "end_line": 20000}, ToolContext())
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
        self.assertIn(os.path.realpath(os.path.join(d, "a.py")), out)
        self.assertIn(os.path.realpath(os.path.join(d, "sub", "b.py")), out)
        out1 = GlobTool().run({"pattern": "*.py", "path": d, "depth": 1}, self.ctx)
        self.assertIn(os.path.realpath(os.path.join(d, "a.py")), out1)
        self.assertNotIn(os.path.realpath(os.path.join(d, "sub", "b.py")), out1)

    @unittest.skipUnless(shutil.which("git"), "git not available")
    def test_glob_and_grep_use_git_backend_in_repo(self):
        repo = self._mkdir("repo")
        subprocess.run(["git", "init", "-q", repo], check=True)
        for name, content in (("a.py", "hello world\n"), ("b.txt", "nope\n")):
            with open(os.path.join(repo, name), "w") as f:
                f.write(content)
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        out = GlobTool().run({"pattern": "*", "path": repo}, self.ctx)
        self.assertIn(os.path.realpath(os.path.join(repo, "a.py")), out)
        self.assertIn(os.path.realpath(os.path.join(repo, "b.txt")), out)
        out = Grep().run({"regex": "hello", "path": repo}, self.ctx)
        self.assertIn("a.py", out)
        self.assertIn("hello", out)
        self.assertNotIn("b.txt", out)
        out = Grep().run({"regex": "hello", "path": repo, "context_lines": 2}, self.ctx)
        self.assertIn("a.py", out)
        self.assertIn("hello", out)

    @unittest.skipUnless(shutil.which("git"), "git not available")
    def test_glob_and_grep_work_through_symlinked_path(self):
        """Regression: a search path that goes through a symlink (macOS
        /var -> /private/var) must be resolved before the git backend
        computes relpaths, or git rejects the pathspec as outside the
        repository."""
        real = self._mkdir("real")
        link = os.path.join(self.tmp.name, "link")
        os.symlink(real, link)
        subprocess.run(["git", "init", "-q", real], check=True)
        with open(os.path.join(real, "a.py"), "w") as f:
            f.write("hello world\n")
        subprocess.run(["git", "add", "."], cwd=real, check=True)
        out = GlobTool().run({"pattern": "*", "path": link}, self.ctx)
        self.assertNotIn("outside repository", out)
        self.assertNotIn("Glob failed", out)
        self.assertIn(os.path.realpath(os.path.join(real, "a.py")), out)
        out = Grep().run({"regex": "hello", "path": link}, self.ctx)
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
        out = Grep().run({"regex": "x", "path": os.path.join(self.tmp.name, "nope")}, self.ctx)
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
            result = Edit().run({"path": path, "old_str": "hello", "new_str": "goodbye"}, ctx)
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
            diff = "--- a/f.txt\n+++ b/f.txt\n@@ -1,3 +1,3 @@\n line1\n-line2\n+lineTWO\n line3\n"
            result = edit_tool().run({"path": path, "new_str": diff, "diff": True}, ctx)
            self.assertIn("Diff successfully applied", result)
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
            bad_diff = "--- a/f.txt\n+++ b/f.txt\n@@ -1,2 +1,2 @@\n line1\n-NOPE\n+lineTWO\n"
            result = edit_tool().run({"path": path, "new_str": bad_diff, "diff": True}, ctx)
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
            result = Edit().run({"path": path, "old_str": "missing", "new_str": "x"}, ctx)
            self.assertIn("Could not find", result)

    def test_old_str_not_unique(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "f.txt")
            with open(path, "w") as f:
                f.write("dup\ndup\n")
            ctx, _ = make_ctx()
            result = Edit().run({"path": path, "old_str": "dup", "new_str": "x"}, ctx)
            self.assertIn("not unique", result)

    def test_string_mode_rejects_directory(self):
        # String replacement is single-file only (mirrors gptel).
        with tempfile.TemporaryDirectory() as d:
            ctx, _ = make_ctx()
            result = Edit().run({"path": d, "old_str": "x", "new_str": "y"}, ctx)
            self.assertIn("intended for single files, not directories", result)

    def test_old_str_takes_precedence_over_diff_flag(self):
        # gptel: string mode when `old_str` is provided, even if diff=True.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "f.txt")
            with open(path, "w") as f:
                f.write("line1\nline2\nline3\n")
            ctx, _ = make_ctx()
            result = Edit().run(
                {"path": path, "old_str": "line2", "new_str": "X", "diff": True},
                ctx,
            )
            self.assertIn("Successfully replaced", result)
            with open(path) as f:
                self.assertEqual(f.read(), "line1\nX\nline3\n")

    def test_diff_false_without_old_str_requires_old_str(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "f.txt")
            with open(path, "w") as f:
                f.write("a\n")
            ctx, _ = make_ctx()
            result = Edit().run({"path": path, "new_str": "z", "diff": False}, ctx)
            self.assertIn("old_str is required", result)

    def test_unreadable_path_errors(self):
        with tempfile.TemporaryDirectory() as d:
            ctx, _ = make_ctx()
            result = Edit().run(
                {"path": os.path.join(d, "nope.txt"), "old_str": "a", "new_str": "b"},
                ctx,
            )
            self.assertIn("is not readable", result)

    def test_diff_mode_patch_binary_missing_reports_error(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "f.txt")
            with open(path, "w") as f:
                f.write("a\nb\nc\n")
            ctx, _ = make_ctx()
            diff = "--- a/f.txt\n+++ b/f.txt\n@@ -1 +1 @@\n-a\n+A\n"
            with mock.patch(
                "python_agent_harness.tools.filesystem.shutil.which",
                return_value=None,
            ):
                result = Edit().run({"path": path, "new_str": diff, "diff": True}, ctx)
            self.assertIn('Command "patch" not available', result)
            with open(path) as f:
                self.assertEqual(f.read(), "a\nb\nc\n")  # untouched


class TestWriteTool(unittest.TestCase):
    def test_new_file_shows_all_lines_added(self):
        with tempfile.TemporaryDirectory() as d:
            ctx, sess = make_ctx()
            result = Write().run({"path": d, "filename": "new.txt", "content": "hi\n"}, ctx)
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
            Write().run({"path": d, "filename": "f.txt", "content": "new content\n"}, ctx)
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
        with open(self.path) as f:
            self.assertEqual(f.read(), "a\nb\nX\nc\n")

    def test_insert_at_beginning(self):
        Insert().run({"path": self.path, "line_number": 0, "new_str": "Z"}, ToolContext())
        with open(self.path) as f:
            self.assertEqual(f.read(), "Z\na\nb\nc\n")

    def test_insert_at_end(self):
        Insert().run({"path": self.path, "line_number": -1, "new_str": "Y"}, ToolContext())
        with open(self.path) as f:
            self.assertEqual(f.read(), "a\nb\nc\nY\n")

    def test_insert_adds_missing_trailing_newline(self):
        Insert().run({"path": self.path, "line_number": 1, "new_str": "X"}, ToolContext())
        with open(self.path) as f:
            self.assertEqual(f.read(), "a\nX\nb\nc\n")


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
        out = Mkdir().run({"parent": self.tmp.name, "name": "sub/deep"}, ToolContext())
        self.assertIn("created/verified", out)
        self.assertTrue(os.path.isdir(os.path.join(self.tmp.name, "sub/deep")))

    def test_existing_directory_is_noop(self):
        d = os.path.join(self.tmp.name, "sub")
        os.makedirs(d)
        out = Mkdir().run({"parent": self.tmp.name, "name": "sub"}, ToolContext())
        self.assertIn("created/verified", out)


class TestSpoolDirAndTruncate(unittest.TestCase):
    """_truncate small-text passthrough and _spool_dir candidate selection."""

    def test_truncate_small_text_passes_through(self):
        from python_agent_harness.tools.filesystem import _truncate

        self.assertEqual(_truncate("small"), "small")

    def test_spool_dir_prefers_tmpdir_env(self):
        import python_agent_harness.tools.filesystem as fs

        with mock.patch.dict(os.environ, {"TMPDIR": "/custom/tmp"}, clear=True):
            self.assertEqual(fs._spool_dir(), "/custom/tmp")

    def test_spool_dir_falls_back_to_system_tempdir(self):
        import python_agent_harness.tools.filesystem as fs

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(fs._spool_dir(), os.path.abspath(tempfile.gettempdir()))

    def test_spool_dir_last_resort_slash_tmp(self):
        import python_agent_harness.tools.filesystem as fs

        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch("tempfile.gettempdir", return_value=""),
        ):
            self.assertEqual(fs._spool_dir(), "/tmp")


class TestCleanupSpoolErrors(unittest.TestCase):
    """cleanup_spooled_files must swallow OSError from os.remove."""

    def test_cleanup_ignores_remove_oserror(self):
        import python_agent_harness.tools.filesystem as fs

        with tempfile.NamedTemporaryFile(delete=False) as f:
            p = f.name
        try:
            fs._spooled_files.append(p)
            with mock.patch("os.remove", side_effect=OSError("busy")):
                fs.cleanup_spooled_files()  # must not raise
            self.assertEqual(fs._spooled_files, [])
        finally:
            fs._spooled_files.clear()
            if os.path.exists(p):
                os.remove(p)


class TestReadErrorPaths(unittest.TestCase):
    """Read streaming-open failure (range reads)."""

    def test_range_read_open_oserror_reported(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "f.txt")
            with open(p, "w") as f:
                f.write("a\nb\nc\n")
            with mock.patch("builtins.open", side_effect=OSError("boom")):
                out = Read().run({"file_path": p, "start_line": 1}, ToolContext())
            self.assertIn("Error: cannot read", out)


class TestGlobErrorPaths(unittest.TestCase):
    """Glob git-ls-files / tree error paths and the empty-result branch."""

    def setUp(self):
        self.ctx = ToolContext()
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _git_repo(self) -> str:
        repo = os.path.join(self.tmp.name, "repo")
        os.makedirs(repo)
        subprocess.run(["git", "init", "-q", repo], check=True)
        with open(os.path.join(repo, "a.py"), "w") as f:
            f.write("hello\n")
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        return repo

    def test_git_lsfiles_timeout_reported(self):
        """git ls-files timing out is reported as an error.  The harness
        `--glob` has no git->tree fallback: inside a repo, git is used and
        a git failure surfaces directly."""
        repo = self._git_repo()
        with mock.patch(
            "python_agent_harness.tools.filesystem.subprocess.run",
            side_effect=subprocess.TimeoutExpired("git ls-files", 60),
        ):
            out = GlobTool().run({"pattern": "*.py", "path": repo}, self.ctx)
        self.assertTrue(out.startswith("Error"))
        self.assertIn("timed out", out)

    def test_git_lsfiles_nonzero_exit_reported(self):
        repo = self._git_repo()
        proc = subprocess.CompletedProcess([], returncode=128, stdout="fatal: bad pathspec\n")
        with mock.patch(
            "python_agent_harness.tools.filesystem.subprocess.run",
            return_value=proc,
        ):
            out = GlobTool().run({"pattern": "*.py", "path": repo}, self.ctx)
        self.assertIn("Glob failed with exit code 128", out)

    def test_git_lsfiles_oserror_reported(self):
        """git ls-files raising OSError is reported (no silent tree
        fallback), mirroring the harness which uses git inside a repo."""
        repo = self._git_repo()
        with mock.patch(
            "python_agent_harness.tools.filesystem.subprocess.run",
            side_effect=OSError("No such file or directory"),
        ):
            out = GlobTool().run({"pattern": "*.py", "path": repo}, self.ctx)
        self.assertTrue(out.startswith("Error"))
        self.assertIn("No such file", out)

    def test_tree_nonzero_exit_reported(self):
        d = os.path.join(self.tmp.name, "plain")
        os.makedirs(d)
        proc = subprocess.CompletedProcess([], returncode=1, stdout="tree stderr\n")
        with (
            mock.patch("shutil.which", return_value="/usr/bin/tree"),
            mock.patch(
                "python_agent_harness.tools.filesystem.subprocess.run",
                return_value=proc,
            ),
        ):
            out = GlobTool().run({"pattern": "*", "path": d}, self.ctx)
        self.assertIn("Glob failed with exit code 1", out)

    def test_git_glob_results_empty_returns_empty_string(self):
        from python_agent_harness.tools.filesystem import _git_glob_results

        self.assertEqual(_git_glob_results("", "/repo", "/repo", None), "")


class TestGrepFallbackBranches(unittest.TestCase):
    """Grep git/rg/grep backend error paths and the rg fallback."""

    def setUp(self):
        self.ctx = ToolContext()
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_git_grep_error_falls_to_unavailable_error(self):
        repo = os.path.join(self.tmp.name, "repo")
        os.makedirs(repo)
        subprocess.run(["git", "init", "-q", repo], check=True)
        with open(os.path.join(repo, "a.py"), "w") as f:
            f.write("hello\n")
        with (
            mock.patch(
                "python_agent_harness.tools.filesystem.subprocess.run",
                side_effect=OSError("git missing"),
            ),
            mock.patch("shutil.which", return_value=None),
        ):
            out = Grep().run({"regex": "hello", "path": repo}, self.ctx)
        self.assertIn("ripgrep/grep/git-grep not available", out)

    def test_rg_fallback_success_with_context_and_glob(self):
        d = os.path.join(self.tmp.name, "plain")
        os.makedirs(d)
        proc = subprocess.CompletedProcess([], returncode=0, stdout="f.txt:1:needle\n")
        with (
            mock.patch("shutil.which", return_value="/usr/bin/rg"),
            mock.patch(
                "python_agent_harness.tools.filesystem.subprocess.run",
                return_value=proc,
            ),
        ):
            out = Grep().run(
                {"regex": "needle", "path": d, "glob": "*.py", "context_lines": 2},
                self.ctx,
            )
        self.assertIn("f.txt:1:needle", out)
        self.assertNotIn("Error", out)

    def test_rg_fallback_error_then_grep_unavailable(self):
        d = os.path.join(self.tmp.name, "plain")
        os.makedirs(d)
        with (
            mock.patch(
                "shutil.which",
                side_effect=lambda n: "/usr/bin/rg" if n == "rg" else None,
            ),
            mock.patch(
                "python_agent_harness.tools.filesystem.subprocess.run",
                side_effect=OSError("boom"),
            ),
        ):
            out = Grep().run({"regex": "x", "path": d}, self.ctx)
        self.assertIn("ripgrep/grep/git-grep not available", out)

    def test_grep_fallback_error_then_unavailable(self):
        d = os.path.join(self.tmp.name, "plain")
        os.makedirs(d)
        with (
            mock.patch(
                "shutil.which",
                side_effect=lambda n: "/usr/bin/grep" if n == "grep" else None,
            ),
            mock.patch(
                "python_agent_harness.tools.filesystem.subprocess.run",
                side_effect=OSError("boom"),
            ),
        ):
            out = Grep().run({"regex": "x", "path": d}, self.ctx)
        self.assertIn("ripgrep/grep/git-grep not available", out)

    def test_git_grep_with_glob_filter(self):
        repo = os.path.join(self.tmp.name, "repo")
        os.makedirs(repo)
        subprocess.run(["git", "init", "-q", repo], check=True)
        with open(os.path.join(repo, "a.py"), "w") as f:
            f.write("needle here\n")
        with open(os.path.join(repo, "b.md"), "w") as f:
            f.write("needle here\n")
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        out = Grep().run({"regex": "needle", "path": repo, "glob": "*.py"}, self.ctx)
        self.assertIn("a.py", out)
        self.assertNotIn("b.md", out)


class TestWriteEditInsertMkdirErrors(unittest.TestCase):
    """OSError paths in Mkdir/Write/Edit/Insert (read and write sides)."""

    def setUp(self):
        self.ctx = ToolContext()
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_mkdir_oserror_reported(self):
        with mock.patch("os.makedirs", side_effect=PermissionError("denied")):
            out = Mkdir().run({"parent": self.tmp.name, "name": "sub"}, self.ctx)
        self.assertIn("Error", out)

    def test_write_existing_unreadable_falls_back_to_blank(self):
        p = os.path.join(self.tmp.name, "f.txt")
        with open(p, "w") as f:
            f.write("secret\n")
        real_open = open

        def fake_open(path, mode="r", *args, **kwargs):
            if "r" in mode:
                raise OSError("permission denied")
            return real_open(path, mode, *args, **kwargs)

        with mock.patch("builtins.open", side_effect=fake_open):
            out = Write().run(
                {"path": self.tmp.name, "filename": "f.txt", "content": "new\n"},
                self.ctx,
            )
        self.assertIn("Created file", out)
        with open(p) as f:
            self.assertEqual(f.read(), "new\n")

    def test_write_open_failure_reported(self):
        with mock.patch("builtins.open", side_effect=OSError("disk full")):
            out = Write().run(
                {"path": self.tmp.name, "filename": "never.txt", "content": "x"},
                self.ctx,
            )
        self.assertIn("Error", out)

    def test_edit_read_oserror_reported(self):
        p = os.path.join(self.tmp.name, "f.txt")
        with open(p, "w") as f:
            f.write("a\n")
        with mock.patch("builtins.open", side_effect=OSError("boom")):
            out = Edit().run({"path": p, "old_str": "a", "new_str": "b"}, self.ctx)
        self.assertIn("Error: cannot read", out)

    def test_edit_missing_old_str_rejected(self):
        # No old_str and no diff flag: gptel routes this to diff mode
        # (old_str nil, diff not false), where `patch` rejects the
        # non-diff `new_str` as garbage.
        p = os.path.join(self.tmp.name, "f.txt")
        with open(p, "w") as f:
            f.write("a\n")
        out = Edit().run({"path": p, "new_str": "b"}, self.ctx)
        self.assertTrue(out.startswith("Error"))
        with open(p) as f:
            self.assertEqual(f.read(), "a\n")  # file untouched

    def test_edit_write_oserror_reported(self):
        p = os.path.join(self.tmp.name, "f.txt")
        with open(p, "w") as f:
            f.write("a\n")
        real_open = open

        def fake_open(path, mode="r", *args, **kwargs):
            if "w" in mode:
                raise OSError("disk full")
            return real_open(path, mode, *args, **kwargs)

        with mock.patch("builtins.open", side_effect=fake_open):
            out = Edit().run({"path": p, "old_str": "a", "new_str": "b"}, self.ctx)
        self.assertIn("Error: disk full", out)

    def test_insert_read_oserror_reported(self):
        p = os.path.join(self.tmp.name, "f.txt")
        with open(p, "w") as f:
            f.write("a\n")
        with mock.patch("builtins.open", side_effect=OSError("boom")):
            out = Insert().run({"path": p, "line_number": 0, "new_str": "x"}, self.ctx)
        self.assertIn("Error: cannot read", out)

    def test_insert_write_oserror_reported(self):
        p = os.path.join(self.tmp.name, "f.txt")
        with open(p, "w") as f:
            f.write("a\n")
        real_open = open

        def fake_open(path, mode="r", *args, **kwargs):
            if "w" in mode:
                raise OSError("disk full")
            return real_open(path, mode, *args, **kwargs)

        with mock.patch("builtins.open", side_effect=fake_open):
            out = Insert().run({"path": p, "line_number": 0, "new_str": "x"}, self.ctx)
        self.assertIn("Error: disk full", out)


@unittest.skipUnless(shutil.which("patch"), "patch not available")
class TestEditDiffModePatch(unittest.TestCase):
    """Diff mode end-to-end via the real `patch` binary (mirrors the diff
    branch of `gptel-agent--edit-files`)."""

    def test_fenced_diff_applies(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "f.txt")
            with open(path, "w") as f:
                f.write("a\nb\nc\n")
            ctx, _ = make_ctx()
            diff = "```diff\n--- a/f.txt\n+++ b/f.txt\n@@ -1,3 +1,3 @@\n a\n-b\n+B\n c\n```\n"
            result = edit_tool().run({"path": path, "new_str": diff, "diff": True}, ctx)
            self.assertIn("Diff successfully applied", result)
            with open(path) as f:
                self.assertEqual(f.read(), "a\nB\nc\n")

    def test_wrong_hunk_counts_are_fixed_and_apply(self):
        # Header counts are deliberately wrong; _fix_patch_headers corrects
        # them so `patch` accepts the hunk.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "f.txt")
            with open(path, "w") as f:
                f.write("a\nb\nc\n")
            ctx, _ = make_ctx()
            diff = "--- a/f.txt\n+++ b/f.txt\n@@ -1,9 +1,9 @@\n a\n-b\n+B\n c\n"
            result = edit_tool().run({"path": path, "new_str": diff, "diff": True}, ctx)
            self.assertIn("Diff successfully applied", result)
            with open(path) as f:
                self.assertEqual(f.read(), "a\nB\nc\n")

    def test_dashed_content_lines_apply(self):
        # Removed/added lines whose content starts with --/++ must be
        # counted as hunk body (not skipped as file headers) or `patch`
        # rejects the diff.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "f.txt")
            with open(path, "w") as f:
                f.write("a\n--removed\nzzz\nc\n")
            ctx, _ = make_ctx()
            diff = "--- a/f.txt\n+++ b/f.txt\n@@ -1,9 +1,9 @@\n a\n---removed\n+++added\n c\n"
            result = edit_tool().run({"path": path, "new_str": diff, "diff": True}, ctx)
            self.assertIn("Diff successfully applied", result)
            with open(path) as f:
                self.assertEqual(f.read(), "a\n++added\nzzz\nc\n")

    def test_dashed_content_line_ending_a_hunk_applies(self):
        # A removed line whose content starts with "--" as the LAST body
        # line of a hunk, with a second hunk following: the header must
        # keep its counts or `patch` rejects the diff as malformed.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "schema.sql")
            with open(path, "w") as f:
                f.write("CREATE TABLE t (\n  id INT\n);\n-- old comment\nSELECT 1;\n-- tail\n")
            ctx, _ = make_ctx()
            diff = (
                "--- a/schema.sql\n+++ b/schema.sql\n"
                "@@ -2,3 +2,2 @@\n   id INT\n );\n--- old comment\n"
                "@@ -6,1 +6,1 @@\n--- tail\n+-- new tail\n"
            )
            result = edit_tool().run({"path": path, "new_str": diff, "diff": True}, ctx)
            self.assertIn("Diff successfully applied", result)
            with open(path) as f:
                self.assertEqual(
                    f.read(), "CREATE TABLE t (\n  id INT\n);\nSELECT 1;\n-- new tail\n"
                )

    def test_directory_multifile_diff_applies(self):
        """A directory path (with trailing slash) + a multi-file unified
        diff edits several files at once, mirroring gptel's directory
        edit mode (patch runs in the directory itself)."""
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "f1.txt"), "w") as f:
                f.write("one\n")
            with open(os.path.join(d, "f2.txt"), "w") as f:
                f.write("two\n")
            ctx, _ = make_ctx()
            diff = (
                "--- a/f1.txt\n+++ b/f1.txt\n@@ -1 +1 @@\n-one\n+ONE\n"
                "--- a/f2.txt\n+++ b/f2.txt\n@@ -1 +1 @@\n-two\n+TWO\n"
            )
            # Trailing slash -> patch runs inside the directory.
            result = edit_tool().run({"path": d + os.sep, "new_str": diff, "diff": True}, ctx)
            self.assertIn("Diff successfully applied", result)
            with open(os.path.join(d, "f1.txt")) as f:
                self.assertEqual(f.read(), "ONE\n")
            with open(os.path.join(d, "f2.txt")) as f:
                self.assertEqual(f.read(), "TWO\n")


class TestEditMac(unittest.TestCase):
    """The macOS Edit backend (pure-Python diff applier) must accept the
    same model diffs that GNU patch accepts on Linux.  Run on every
    platform so the macOS path is covered by Linux CI too."""

    def _file(self, d: str, name: str, content: str) -> str:
        path = os.path.join(d, name)
        with open(path, "w") as f:
            f.write(content)
        return path

    def test_dashed_content_line_ending_a_hunk_applies(self):
        """The macOS CI failure: a removed line starting with -- as the
        LAST body line of a hunk, followed by a second hunk.  Apple's
        BSD patch rejects it; the Python applier must apply it."""
        with tempfile.TemporaryDirectory() as d:
            path = self._file(
                d,
                "schema.sql",
                "CREATE TABLE t (\n  id INT\n);\n-- old comment\nSELECT 1;\n-- tail\n",
            )
            ctx, sess = make_ctx()
            diff = (
                "--- a/schema.sql\n+++ b/schema.sql\n"
                "@@ -2,3 +2,2 @@\n   id INT\n );\n--- old comment\n"
                "@@ -6,1 +6,1 @@\n--- tail\n+-- new tail\n"
            )
            result = EditMac().run({"path": path, "new_str": diff, "diff": True}, ctx)
            self.assertIn("Diff successfully applied", result)
            with open(path) as f:
                self.assertEqual(
                    f.read(), "CREATE TABLE t (\n  id INT\n);\nSELECT 1;\n-- new tail\n"
                )
            self.assertEqual(len(sess.recorded_diffs), 1)

    def test_simple_replace_applies(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._file(d, "f.txt", "line1\nline2\nline3\n")
            ctx, sess = make_ctx()
            diff = "--- a/f.txt\n+++ b/f.txt\n@@ -1,3 +1,3 @@\n line1\n-line2\n+lineTWO\n line3\n"
            result = EditMac().run({"path": path, "new_str": diff, "diff": True}, ctx)
            self.assertIn("Diff successfully applied", result)
            with open(path) as f:
                self.assertEqual(f.read(), "line1\nlineTWO\nline3\n")
            self.assertEqual(len(sess.recorded_diffs), 1)

    def test_wrong_hunk_counts_recounted(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._file(d, "f.txt", "a\nb\nc\n")
            ctx, _ = make_ctx()
            diff = "--- a/f.txt\n+++ b/f.txt\n@@ -1,9 +1,9 @@\n a\n-b\n+B\n c\n"
            result = EditMac().run({"path": path, "new_str": diff, "diff": True}, ctx)
            self.assertIn("Diff successfully applied", result)
            with open(path) as f:
                self.assertEqual(f.read(), "a\nB\nc\n")

    def test_fenced_diff_applies(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._file(d, "f.txt", "a\nb\nc\n")
            ctx, _ = make_ctx()
            diff = "```diff\n--- a/f.txt\n+++ b/f.txt\n@@ -1,3 +1,3 @@\n a\n-b\n+B\n c\n```\n"
            result = EditMac().run({"path": path, "new_str": diff, "diff": True}, ctx)
            self.assertIn("Diff successfully applied", result)
            with open(path) as f:
                self.assertEqual(f.read(), "a\nB\nc\n")

    def test_dashed_content_lines_apply(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._file(d, "f.txt", "a\n--removed\nzzz\nc\n")
            ctx, _ = make_ctx()
            diff = "--- a/f.txt\n+++ b/f.txt\n@@ -1,9 +1,9 @@\n a\n---removed\n+++added\n c\n"
            result = EditMac().run({"path": path, "new_str": diff, "diff": True}, ctx)
            self.assertIn("Diff successfully applied", result)
            with open(path) as f:
                self.assertEqual(f.read(), "a\n++added\nzzz\nc\n")

    def test_mismatch_errors_and_leaves_file_untouched(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._file(d, "f.txt", "line1\nline2\nline3\n")
            ctx, sess = make_ctx()
            bad_diff = "--- a/f.txt\n+++ b/f.txt\n@@ -1,2 +1,2 @@\n line1\n-NOPE\n+lineTWO\n"
            result = EditMac().run({"path": path, "new_str": bad_diff, "diff": True}, ctx)
            self.assertTrue(result.startswith("Error:"))
            with open(path) as f:
                self.assertEqual(f.read(), "line1\nline2\nline3\n")
            self.assertEqual(sess.recorded_diffs, [])

    def test_directory_multifile_diff_applies(self):
        with tempfile.TemporaryDirectory() as d:
            self._file(d, "f1.txt", "one\n")
            self._file(d, "f2.txt", "two\n")
            ctx, _ = make_ctx()
            diff = (
                "--- a/f1.txt\n+++ b/f1.txt\n@@ -1 +1 @@\n-one\n+ONE\n"
                "--- a/f2.txt\n+++ b/f2.txt\n@@ -1 +1 @@\n-two\n+TWO\n"
            )
            result = EditMac().run({"path": d + os.sep, "new_str": diff, "diff": True}, ctx)
            self.assertIn("Diff successfully applied", result)
            with open(os.path.join(d, "f1.txt")) as f:
                self.assertEqual(f.read(), "ONE\n")
            with open(os.path.join(d, "f2.txt")) as f:
                self.assertEqual(f.read(), "TWO\n")

    def test_insert_only_hunk(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._file(d, "f.txt", "a\nc\n")
            ctx, _ = make_ctx()
            diff = "--- a/f.txt\n+++ b/f.txt\n@@ -1,1 +1,2 @@\n a\n+b\n c\n"
            result = EditMac().run({"path": path, "new_str": diff, "diff": True}, ctx)
            self.assertIn("Diff successfully applied", result)
            with open(path) as f:
                self.assertEqual(f.read(), "a\nb\nc\n")

    def test_fuzz_matches_context_one_line_off(self):
        """A context line that does not match (model drift) still
        applies when the removed lines match, up to the fuzz limit."""
        with tempfile.TemporaryDirectory() as d:
            path = self._file(d, "f.txt", "x\na\nzzz\nc\n")
            ctx, _ = make_ctx()
            diff = "--- a/f.txt\n+++ b/f.txt\n@@ -1,3 +1,3 @@\n a\n-zzz\n+c\n"
            result = EditMac().run({"path": path, "new_str": diff, "diff": True}, ctx)
            self.assertIn("Diff successfully applied", result)
            with open(path) as f:
                self.assertEqual(f.read(), "x\na\nc\nc\n")

    def test_no_newline_at_eof_handled(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._file(d, "f.txt", "a\nb")
            ctx, _ = make_ctx()
            diff = "--- a/f.txt\n+++ b/f.txt\n@@ -1,2 +1,2 @@\n a\n-b\n+b\n\\ No newline at end of file\n"
            result = EditMac().run({"path": path, "new_str": diff, "diff": True}, ctx)
            self.assertIn("Diff successfully applied", result)
            with open(path) as f:
                self.assertEqual(f.read(), "a\nb")


if __name__ == "__main__":
    unittest.main()


class TestGlobMac(unittest.TestCase):
    """GlobMac: pure-Python non-git fallback (pathlib) and git delegation.

    Run on every platform so the macOS path is covered by Linux CI too
    (same approach as TestEditMac).
    """

    def setUp(self):
        self.ctx = ToolContext()
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _mkdir(self, *parts) -> str:
        p = os.path.join(self.tmp.name, *parts)
        os.makedirs(p, exist_ok=True)
        return p

    def _file(self, *parts, content: str = "") -> str:
        p = os.path.join(self.tmp.name, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(content)
        return p

    def test_pathlib_fallback_lists_files(self):
        """Non-git directory: GlobMac uses pathlib instead of tree."""
        from python_agent_harness.tools.glob_mac import GlobMac

        d = self._mkdir("proj")
        self._file("proj", "a.py")
        self._file("proj", "b.txt")
        out = GlobMac().run({"pattern": "*.py", "path": d}, self.ctx)
        self.assertIn(os.path.realpath(os.path.join(d, "a.py")), out)
        self.assertNotIn("b.txt", out)

    def test_pathlib_fallback_depth_limiting(self):
        """depth=1 excludes files in subdirectories."""
        from python_agent_harness.tools.glob_mac import GlobMac

        d = self._mkdir("proj")
        self._file("proj", "top.py")
        self._file("proj", "sub", "deep.py")
        out = GlobMac().run({"pattern": "*.py", "path": d, "depth": 1}, self.ctx)
        self.assertIn(os.path.realpath(os.path.join(d, "top.py")), out)
        self.assertNotIn("deep.py", out)

    def test_pathlib_fallback_unlimited_depth(self):
        """Without depth, files at any level are returned."""
        from python_agent_harness.tools.glob_mac import GlobMac

        d = self._mkdir("proj")
        self._file("proj", "top.py")
        self._file("proj", "sub", "deep.py")
        out = GlobMac().run({"pattern": "*.py", "path": d}, self.ctx)
        self.assertIn(os.path.realpath(os.path.join(d, "top.py")), out)
        self.assertIn(os.path.realpath(os.path.join(d, "sub", "deep.py")), out)

    def test_pathlib_fallback_skips_hidden_dirs(self):
        """Dotfiles/directories (e.g. .git) are excluded from results."""
        from python_agent_harness.tools.glob_mac import GlobMac

        d = self._mkdir("proj")
        self._file("proj", "visible.py")
        self._file("proj", ".hidden", "secret.py")
        out = GlobMac().run({"pattern": "*.py", "path": d}, self.ctx)
        self.assertIn("visible.py", out)
        self.assertNotIn("secret.py", out)

    def test_pathlib_fallback_case_insensitive(self):
        """Glob matching is case-insensitive (mirrors tree --ignore-case)."""
        from python_agent_harness.tools.glob_mac import GlobMac

        d = self._mkdir("proj")
        self._file("proj", "README.PY")
        out = GlobMac().run({"pattern": "*.py", "path": d}, self.ctx)
        self.assertIn("README.PY", out)

    def test_pathlib_fallback_no_matches_returns_empty(self):
        """No matching files returns empty string."""
        from python_agent_harness.tools.glob_mac import GlobMac

        d = self._mkdir("proj")
        self._file("proj", "a.txt")
        out = GlobMac().run({"pattern": "*.rs", "path": d}, self.ctx)
        self.assertEqual(out, "")

    @unittest.skipUnless(shutil.which("git"), "git not available")
    def test_git_delegation(self):
        """Inside a git repo, GlobMac delegates to the parent (git ls-files)."""
        from python_agent_harness.tools.glob_mac import GlobMac

        repo = self._mkdir("repo")
        subprocess.run(["git", "init", "-q", repo], check=True)
        self._file("repo", "a.py", content="hello\n")
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        out = GlobMac().run({"pattern": "*", "path": repo}, self.ctx)
        self.assertIn(os.path.realpath(os.path.join(repo, "a.py")), out)

    def test_empty_pattern_errors(self):
        from python_agent_harness.tools.glob_mac import GlobMac

        out = GlobMac().run({"pattern": "", "path": self.tmp.name}, self.ctx)
        self.assertIn("Error", out)

    def test_nonexistent_path_errors(self):
        from python_agent_harness.tools.glob_mac import GlobMac

        out = GlobMac().run({"pattern": "*", "path": os.path.join(self.tmp.name, "nope")}, self.ctx)
        self.assertIn("Error", out)

    def test_pathlib_fallback_sorted_by_mtime(self):
        """Results are sorted by mtime, newest first."""
        import time

        from python_agent_harness.tools.glob_mac import GlobMac

        d = self._mkdir("proj")
        older = self._file("proj", "older.py")
        time.sleep(0.05)
        newer = self._file("proj", "newer.py")
        out = GlobMac().run({"pattern": "*.py", "path": d}, self.ctx)
        older_pos = out.index(os.path.realpath(older))
        newer_pos = out.index(os.path.realpath(newer))
        self.assertLess(newer_pos, older_pos, "newer file should appear first")


class TestGrepMac(unittest.TestCase):
    """GrepMac: git grep -E instead of -P, plus fallback delegation.

    Run on every platform so the macOS path is covered by Linux CI too.
    """

    def setUp(self):
        self.ctx = ToolContext()
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _mkdir(self, *parts) -> str:
        p = os.path.join(self.tmp.name, *parts)
        os.makedirs(p, exist_ok=True)
        return p

    @unittest.skipUnless(shutil.which("git"), "git not available")
    def test_git_grep_uses_extended_regex(self):
        """GrepMac uses git grep -E (not -P), which works on stock macOS."""
        from python_agent_harness.tools.grep_mac import GrepMac

        repo = self._mkdir("repo")
        subprocess.run(["git", "init", "-q", repo], check=True)
        with open(os.path.join(repo, "a.py"), "w") as f:
            f.write("hello world\n")
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        out = GrepMac().run({"regex": "hello", "path": repo}, self.ctx)
        self.assertIn("a.py", out)
        self.assertIn("hello", out)

    @unittest.skipUnless(shutil.which("git"), "git not available")
    def test_git_grep_e_flag_in_command(self):
        """Verify the actual command uses -E, not -P."""
        from python_agent_harness.tools.grep_mac import GrepMac

        repo = self._mkdir("repo")
        subprocess.run(["git", "init", "-q", repo], check=True)
        with open(os.path.join(repo, "a.py"), "w") as f:
            f.write("hello\n")
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        with mock.patch(
            "python_agent_harness.tools.grep_mac.subprocess.run",
            wraps=subprocess.run,
        ) as spy:
            GrepMac().run({"regex": "hello", "path": repo}, self.ctx)
            cmd = spy.call_args_list[0][0][0]
            self.assertIn("-E", cmd)
            self.assertNotIn("-P", cmd)

    @unittest.skipUnless(shutil.which("git"), "git not available")
    def test_git_grep_with_context_lines(self):
        from python_agent_harness.tools.grep_mac import GrepMac

        repo = self._mkdir("repo")
        subprocess.run(["git", "init", "-q", repo], check=True)
        with open(os.path.join(repo, "a.py"), "w") as f:
            f.write("line1\nline2\nhello\nline4\nline5\n")
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        out = GrepMac().run({"regex": "hello", "path": repo, "context_lines": 1}, self.ctx)
        self.assertIn("hello", out)

    @unittest.skipUnless(shutil.which("git"), "git not available")
    def test_git_grep_with_glob_filter(self):
        from python_agent_harness.tools.grep_mac import GrepMac

        repo = self._mkdir("repo")
        subprocess.run(["git", "init", "-q", repo], check=True)
        with open(os.path.join(repo, "a.py"), "w") as f:
            f.write("needle\n")
        with open(os.path.join(repo, "b.md"), "w") as f:
            f.write("needle\n")
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        out = GrepMac().run({"regex": "needle", "path": repo, "glob": "*.py"}, self.ctx)
        self.assertIn("a.py", out)
        self.assertNotIn("b.md", out)

    def test_nonexistent_path_errors(self):
        from python_agent_harness.tools.grep_mac import GrepMac

        out = GrepMac().run({"regex": "x", "path": os.path.join(self.tmp.name, "nope")}, self.ctx)
        self.assertIn("Error", out)

    def test_fallback_to_parent_on_non_git(self):
        """Outside a git repo, GrepMac delegates to the parent's
        rg/grep fallback chain."""
        from python_agent_harness.tools.grep_mac import GrepMac

        d = self._mkdir("plain")
        with open(os.path.join(d, "f.txt"), "w") as f:
            f.write("needle here\n")
        proc = subprocess.CompletedProcess([], returncode=0, stdout="f.txt:1:needle here\n")
        with (
            mock.patch("shutil.which", return_value="/usr/bin/rg"),
            mock.patch(
                "python_agent_harness.tools.grep_mac.subprocess.run",
                return_value=proc,
            ),
        ):
            out = GrepMac().run({"regex": "needle", "path": d}, self.ctx)
        self.assertIn("needle", out)

    @unittest.skipUnless(shutil.which("git"), "git not available")
    def test_git_grep_failure_falls_to_parent(self):
        """When git grep fails (e.g. OSError), GrepMac falls through to
        the parent's rg/grep chain."""
        from python_agent_harness.tools.grep_mac import GrepMac

        repo = self._mkdir("repo")
        subprocess.run(["git", "init", "-q", repo], check=True)
        with open(os.path.join(repo, "a.py"), "w") as f:
            f.write("hello\n")
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        with (
            mock.patch(
                "python_agent_harness.tools.grep_mac.subprocess.run",
                side_effect=OSError("git broke"),
            ),
            mock.patch("shutil.which", return_value=None),
        ):
            out = GrepMac().run({"regex": "hello", "path": repo}, self.ctx)
        self.assertIn("ripgrep/grep/git-grep not available", out)


if __name__ == "__main__":
    unittest.main()
