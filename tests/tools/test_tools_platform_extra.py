"""Extra platform-specific tool tests: grep_mac regex translation,
glob_mac fallback edges, and the edit_mac unreadable-content paths.

Existing tests in test_filesystem.py drive the public ``run()`` flows;
this file reaches the private helpers and failure branches those flows
don't hit (PCRE->ERE translation, rg-present path, OSError tolerance
during traversal/stat/reads), so the platform backends are covered by
CI of every OS — none of these tests depend on the host platform.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from python_agent_harness.tools.base import ToolContext
from python_agent_harness.tools.grep_mac import (
    GrepMac,
    _find_char_class_end,
    _is_escaped,
    _pcre_to_ere,
)


class FakeSession:
    """Runtime double: only cwd + record_diff are reached by these tools."""

    def __init__(self, project_dir: str = "/tmp") -> None:
        self.project_dir = project_dir
        self.recorded_diffs: list[str] = []

    def record_diff(self, diff_text: str) -> None:
        self.recorded_diffs.append(diff_text)


def _ctx(session: FakeSession | None = None) -> ToolContext:
    return ToolContext(session or FakeSession())


class TestPCREHelpers(unittest.TestCase):
    def test_is_escaped_odd_and_even_backslashes(self):
        self.assertTrue(_is_escaped("\\d", 1))  # \d: d is escaped
        self.assertFalse(_is_escaped("\\\\d", 2))  # \\d: literal backslash + d
        self.assertFalse(_is_escaped("d", 0))  # nothing before it
        self.assertTrue(_is_escaped("\\\\\\d", 3))  # \\\d: escaped again

    def test_find_char_class_end_plain(self):
        self.assertEqual(_find_char_class_end("[abc]", 0), 4)

    def test_find_char_class_end_negated(self):
        self.assertEqual(_find_char_class_end("[^]]", 0), 3)

    def test_find_char_class_end_literal_closing_bracket(self):
        self.assertEqual(_find_char_class_end("[]]", 0), 2)

    def test_find_char_class_end_escaped_bracket(self):
        self.assertEqual(_find_char_class_end("[a\\]b]", 0), 5)

    def test_find_char_class_end_unterminated(self):
        self.assertIsNone(_find_char_class_end("[abc", 0))
        self.assertIsNone(_find_char_class_end("[", 0))


class TestPCREToERE(unittest.TestCase):
    def test_digit_shorthand(self):
        self.assertEqual(_pcre_to_ere(r"\d+"), "[0-9]+")

    def test_all_shorthands_map(self):
        self.assertEqual(
            _pcre_to_ere(r"\D\w\W\s\S"),
            "[^0-9][A-Za-z0-9_][^A-Za-z0-9_][ \t\n\r\f\v][^ \t\n\r\f\v]",
        )

    def test_shorthand_inside_class_untouched(self):
        # replacing inside [] would create nested brackets: invalid in ERE
        self.assertEqual(_pcre_to_ere(r"[\d.]"), r"[\d.]")

    def test_escaped_literal_backslash_pair_untouched(self):
        # \\d is a literal backslash followed by 'd', not a shorthand
        self.assertEqual(_pcre_to_ere(r"\\d"), r"\\d")

    def test_word_boundary_at_start(self):
        self.assertEqual(_pcre_to_ere(r"\bdog"), "[[:<:]]dog")

    def test_word_boundary_at_end(self):
        self.assertEqual(_pcre_to_ere(r"dog\b"), "dog[[:>:]]")

    def test_word_boundary_alone(self):
        self.assertEqual(_pcre_to_ere(r"\b"), "[[:<:]]")

    def test_word_boundary_after_nonword_char(self):
        self.assertEqual(_pcre_to_ere(r"(\bfoo"), "([[:<:]]foo")

    def test_inside_unterminated_class_not_treated_as_class(self):
        # no closing ]: not a class span, so the shorthand is translated
        self.assertEqual(_pcre_to_ere("x[\\dy"), "x[[0-9]y")

    def test_lookahead_left_as_is(self):
        self.assertEqual(_pcre_to_ere(r"foo(?=bar)"), r"foo(?=bar)")


class TestGrepMacRun(unittest.TestCase):
    def test_subprocess_oserror_falls_back_to_rg_grep(self):
        """A failed git grep -E run must not propagate: GrepMac falls
        through to the shared rg/grep chain."""
        with (
            tempfile.TemporaryDirectory() as d,
            mock.patch("python_agent_harness.tools.grep_mac._git_root", return_value=d),
            mock.patch(
                "python_agent_harness.tools.grep_mac.subprocess.run",
                side_effect=OSError("git exploded"),
            ),
            mock.patch.object(GrepMac, "_fallback_rg_grep", return_value="fallback output") as fb,
        ):
            out = GrepMac().run({"regex": "x", "path": d}, _ctx())
        self.assertEqual(out, "fallback output")
        self.assertEqual(fb.call_args.args[0], r"x")


class TestGlobMacFallback(unittest.TestCase):
    def test_find_failure_reported(self):
        from python_agent_harness.tools.glob_mac import GlobMac

        with mock.patch(
            "python_agent_harness.tools.glob_mac.subprocess.run",
            side_effect=OSError("no find"),
        ):
            out = GlobMac()._find_fallback("*.py", "/tmp", None)
        self.assertIn("Error: no find", out)

    def test_find_nonzero_exit_reported(self):
        from python_agent_harness.tools.glob_mac import GlobMac

        proc = SimpleNamespace(stdout="", stderr="find: bad\n", returncode=2)
        with mock.patch("python_agent_harness.tools.glob_mac.subprocess.run", return_value=proc):
            out = GlobMac()._find_fallback("*.py", "/tmp", None)
        self.assertIn("Glob failed with exit code 2", out)
        self.assertIn("find: bad", out)

    def test_mtime_lookup_failure_tolerated(self):
        from python_agent_harness.tools.glob_mac import GlobMac

        proc = SimpleNamespace(stdout="a.py\nb.py\n", stderr="", returncode=0)
        with (
            mock.patch("python_agent_harness.tools.glob_mac.subprocess.run", return_value=proc),
            mock.patch(
                "python_agent_harness.tools.glob_mac.os.path.getmtime",
                side_effect=OSError("gone"),
            ),
        ):
            out = GlobMac()._find_fallback("*.py", "/tmp", None)
        self.assertIn("a.py", out)
        self.assertIn("b.py", out)

    def test_depth_limits_find(self):
        from python_agent_harness.tools.glob_mac import GlobMac

        proc = SimpleNamespace(stdout="", stderr="", returncode=0)
        captured: dict = {}

        def fake_run(cmd, **kw):
            captured["cmd"] = cmd
            return proc

        with mock.patch("python_agent_harness.tools.glob_mac.subprocess.run", side_effect=fake_run):
            GlobMac()._find_fallback("*.py", "/tmp", 2)
        self.assertIn("-maxdepth", captured["cmd"])
        self.assertIn("2", captured["cmd"])

    def test_run_without_path_uses_cwd(self):
        from python_agent_harness.tools.glob_mac import GlobMac

        with tempfile.TemporaryDirectory() as d:
            proc = SimpleNamespace(stdout=f"{d}/a.py\n", stderr="", returncode=0)
            with mock.patch(
                "python_agent_harness.tools.glob_mac.subprocess.run", return_value=proc
            ):
                out = GlobMac().run({"pattern": "*.py"}, _ctx(FakeSession(d)))
        self.assertIn("a.py", out)


class TestEditUnreadableContent(unittest.TestCase):
    DIFF = "--- a/f.txt\n+++ b/f.txt\n@@ -1,3 +1,3 @@\n line1\n-line2\n+lineTWO\n line3\n"

    def _file(self, d: str) -> str:
        path = os.path.join(d, "f.txt")
        with open(path, "w") as f:
            f.write("line1\nline2\nline3\n")
        return path

    def _failing_open(self, fail_on: int):
        real_open = open
        calls = {"n": 0}

        def fake_open(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == fail_on:
                raise OSError("denied")
            return real_open(*args, **kwargs)

        return fake_open

    def test_edit_mac_unreadable_old_content_applies(self):
        from python_agent_harness.tools.edit_mac import EditMac

        with tempfile.TemporaryDirectory() as d:
            path = self._file(d)
            ctx = _ctx()
            with (
                mock.patch(
                    "python_agent_harness.tools.edit_mac.apply_unified_diff",
                    return_value=(True, "applied"),
                ),
                mock.patch("builtins.open", side_effect=self._failing_open(1)),
            ):
                result = EditMac().run({"path": path, "new_str": self.DIFF, "diff": True}, ctx)
        self.assertIn("Diff successfully applied", result)

    def test_edit_mac_unreadable_new_content_tolerated(self):
        from python_agent_harness.tools.edit_mac import EditMac

        with tempfile.TemporaryDirectory() as d:
            path = self._file(d)
            ctx = _ctx()
            with (
                mock.patch(
                    "python_agent_harness.tools.edit_mac.apply_unified_diff",
                    return_value=(True, "applied"),
                ),
                mock.patch("builtins.open", side_effect=self._failing_open(2)),
            ):
                result = EditMac().run({"path": path, "new_str": self.DIFF, "diff": True}, ctx)
        self.assertIn("Diff successfully applied", result)


if __name__ == "__main__":
    unittest.main()
