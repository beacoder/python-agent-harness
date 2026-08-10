import unittest

from python_agent_harness.session_store import (
    SessionStore, sanitize_title, title_from_filename,
)


class TestSession(unittest.TestCase):
    def test_sanitize_title(self):
        self.assertEqual(sanitize_title('  "Fix bug"  '), "Fix-bug")
        self.assertEqual(sanitize_title("a/b\\c:d*e?f\"g<h>i|j"), "a-b-c-d-e-f-g-h-i-j")
        self.assertEqual(sanitize_title("  spaced   title  "), "spaced-title")
        self.assertEqual(sanitize_title("x" * 60), "x" * 50)
        self.assertEqual(sanitize_title("multi\nline\rtitle"), "multi-line-title")

    def test_title_from_filename(self):
        self.assertEqual(title_from_filename("/sessions/fix-bug_260805120000.md"), "fix bug")
        self.assertEqual(
            title_from_filename("/sessions/fix-bug_260805120000-1.md"), "fix bug"
        )
        self.assertIsNone(title_from_filename("/sessions/singleword_260805120000.md"))
        self.assertIsNone(title_from_filename("/sessions/plain.md"))

    def test_metadata_roundtrip(self):
        store = SessionStore(
            project_dir="/tmp/proj", model="deepseek-v4", backend="DeepSeek",
            system_prompt="be helpful", temperature=0.7, max_tokens=8192,
            tool_names=["glob", "grep"],
        )
        meta = store.metadata_block()
        text = "conversation...\n\n" + meta + "\n"
        parsed = SessionStore.parse_metadata(text)
        self.assertEqual(parsed["python-agent-harness--project-dir"], "/tmp/proj")
        self.assertEqual(parsed["gptel-model"], "deepseek-v4")
        self.assertIn("glob", parsed["gptel--tool-names"])
        stripped = SessionStore.strip_metadata(text)
        self.assertEqual(stripped.strip(), "conversation...")

    def test_save_and_restore_flow(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            store = SessionStore(
                project_dir=d, model="m", backend="b",
            )
            path = store.save("hello world")
            self.assertTrue(path)
            self.assertTrue(os.path.exists(path))
            latest = SessionStore.latest_session()
            self.assertEqual(latest, path)

    def test_apply_title_renames(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            from python_agent_harness import config

            old_dir = config.SESSION_DIR
            config.SESSION_DIR = __import__("pathlib").Path(d)
            try:
                store = SessionStore(project_dir="/tmp/proj", model="m", backend="b")
                store.save("x")
                store.apply_title("My Great Session")
                self.assertTrue(os.path.exists(store.file_path))
                self.assertEqual(os.path.basename(store.file_path).startswith("My-Great-Session_"), True)
            finally:
                config.SESSION_DIR = old_dir

    def test_same_second_save_collision_keeps_both(self):
        """Two sessions with the same project saved in the same second
        must not overwrite each other (numeric suffix instead)."""
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            from python_agent_harness import config

            old_dir = config.SESSION_DIR
            config.SESSION_DIR = __import__("pathlib").Path(d)
            try:
                s1 = SessionStore(project_dir="/tmp/proj", model="m", backend="b")
                p1 = s1.save("one")
                s2 = SessionStore(project_dir="/tmp/proj", model="m", backend="b")
                p2 = s2.save("two")
                self.assertNotEqual(p1, p2)
                self.assertTrue(os.path.exists(p1))
                self.assertTrue(os.path.exists(p2))
                with open(p1) as f:
                    self.assertIn("one", f.read())
                with open(p2) as f:
                    self.assertIn("two", f.read())
            finally:
                config.SESSION_DIR = old_dir

    def test_title_collision_keeps_both_files(self):
        """Two sessions given the same title in the same second must not
        overwrite each other."""
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            from python_agent_harness import config

            old_dir = config.SESSION_DIR
            config.SESSION_DIR = __import__("pathlib").Path(d)
            try:
                s1 = SessionStore(project_dir="/tmp/proj", model="m", backend="b")
                s1.save("one")
                s1.apply_title("Same Title")
                t1 = s1.file_path
                s2 = SessionStore(project_dir="/tmp/proj", model="m", backend="b")
                s2.save("two")
                s2.apply_title("Same Title")
                t2 = s2.file_path
                self.assertNotEqual(t1, t2)
                self.assertTrue(os.path.exists(t1))
                self.assertTrue(os.path.exists(t2))
            finally:
                config.SESSION_DIR = old_dir

    def test_save_returns_none_without_file_path(self):
        """Defensive path: if session_file() cannot produce a path, save
        must return None instead of writing anywhere."""
        import unittest.mock as mock

        store = SessionStore(project_dir="/tmp/proj", model="m", backend="b")
        with mock.patch.object(SessionStore, "session_file", return_value=None):
            self.assertIsNone(store.save("hello"))

    def test_apply_title_requires_nonempty_sanitized_title(self):
        """A title that sanitizes to '' must not rename anything and must
        not be recorded."""
        store = SessionStore(project_dir="/tmp/proj", model="m", backend="b")
        store.apply_title("   ---   ")  # sanitizes to ""
        self.assertIsNone(store.title)
        self.assertIsNone(store.file_path)

    def test_apply_title_oserror_keeps_original_file(self):
        """A failed rename (OSError) must keep the original file and not
        record the title (the pending title retries on the next save)."""
        import os
        import tempfile
        import unittest.mock as mock

        with tempfile.TemporaryDirectory() as d:
            from python_agent_harness import config

            old_dir = config.SESSION_DIR
            config.SESSION_DIR = __import__("pathlib").Path(d)
            try:
                store = SessionStore(project_dir="/tmp/proj", model="m", backend="b")
                store.save("x")
                before = store.file_path
                with mock.patch(
                    "python_agent_harness.session_store.os.replace",
                    side_effect=OSError("permission denied"),
                ):
                    store.apply_title("New Title")
                self.assertEqual(store.file_path, before)
                self.assertIsNone(store.title)
                self.assertTrue(os.path.exists(before))
            finally:
                config.SESSION_DIR = old_dir

    def test_parse_metadata_without_block(self):
        self.assertEqual(SessionStore.parse_metadata("just a conversation"), {})

    def test_parse_metadata_skips_foreign_lines_and_unparsable_values(self):
        text = (
            "conversation\n"
            ";; Local Variables:\n"
            "plain line without marker\n"
            ";; no-colon-here\n"
            ";; gptel-model: 'm'\n"
            ";; gptel--tool-names: ['a', 'b']\n"
            ";; gptel-system-prompt: unquoted-value\n"
            ";; End:\n"
        )
        parsed = SessionStore.parse_metadata(text)
        self.assertEqual(parsed["gptel-model"], "m")
        self.assertEqual(parsed["gptel--tool-names"], "a b")
        self.assertEqual(parsed["gptel-system-prompt"], "unquoted-value")
        self.assertNotIn("plain line without marker", parsed)

    def test_strip_metadata_without_block(self):
        self.assertEqual(SessionStore.strip_metadata("plain"), "plain")

    def test_strip_metadata_without_end_marker(self):
        text = "conversation\n;; Local Variables:\n;; gptel-model: 'm'\n"
        self.assertEqual(SessionStore.strip_metadata(text), "conversation")

    def test_latest_session_missing_dir(self):
        import tempfile
        from pathlib import Path

        from python_agent_harness import config

        with tempfile.TemporaryDirectory() as d:
            old_dir = config.SESSION_DIR
            config.SESSION_DIR = Path(d) / "does-not-exist"
            try:
                self.assertIsNone(SessionStore.latest_session())
            finally:
                config.SESSION_DIR = old_dir

    def test_list_sessions_newest_first(self):
        import os
        import tempfile
        from pathlib import Path

        from python_agent_harness import config

        with tempfile.TemporaryDirectory() as d:
            old_dir = config.SESSION_DIR
            config.SESSION_DIR = Path(d)
            try:
                s1 = SessionStore(project_dir="/tmp/proj", model="m", backend="b")
                p1 = s1.save("one")
                s2 = SessionStore(project_dir="/tmp/proj", model="m", backend="b")
                p2 = s2.save("two")
                os.utime(p1, (1_000_000, 1_000_000))  # make p1 the older file
                sessions = SessionStore.list_sessions()
                self.assertEqual(sessions, [p2, p1])
            finally:
                config.SESSION_DIR = old_dir

    def test_list_sessions_missing_dir(self):
        import tempfile
        from pathlib import Path

        from python_agent_harness import config

        with tempfile.TemporaryDirectory() as d:
            old_dir = config.SESSION_DIR
            config.SESSION_DIR = Path(d) / "nope"
            try:
                self.assertEqual(SessionStore.list_sessions(), [])
            finally:
                config.SESSION_DIR = old_dir


if __name__ == "__main__":
    unittest.main()
