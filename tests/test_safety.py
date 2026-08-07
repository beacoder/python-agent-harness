import os
import tempfile
import unittest

from python_agent_harness.safety import (
    BashPolicy, SafetyViolation, bash_read_only_p, check_path,
    command_forbidden, path_forbidden,
)
from python_agent_harness.undo import UndoStack


class TestPathSafety(unittest.TestCase):
    def test_forbidden_mnt(self):
        self.assertIsNotNone(path_forbidden("/mnt/data"))
        self.assertIsNone(path_forbidden("/home/user/file.py"))

    def test_forbidden_path_canonicalized(self):
        """Syntactic bypasses (//, /../, symlinks) must not evade the
        forbidden-path patterns; the bare directory is caught too."""
        self.assertIsNotNone(path_forbidden("//mnt/data"))
        self.assertIsNotNone(path_forbidden("/tmp/../mnt/data"))
        self.assertIsNotNone(path_forbidden("/mnt"))  # bare dir via trailing sep
        self.assertIsNone(path_forbidden("/mnt2/data"))
        self.assertIsNone(path_forbidden("/home/user/file.py"))

    def test_check_path_raises(self):
        with self.assertRaises(SafetyViolation):
            check_path("/mnt/data", "Read")
        check_path("/home/user/file.py", "Read")  # no raise

    def test_command_forbidden(self):
        self.assertIsNotNone(command_forbidden("cat /mnt/secret.txt"))
        self.assertIsNone(command_forbidden("ls /home"))

    def test_command_forbidden_canonicalized(self):
        self.assertIsNotNone(command_forbidden("ls //mnt/data"))
        self.assertIsNotNone(command_forbidden("ls /tmp/../mnt/data"))
        self.assertIsNotNone(command_forbidden("ls /mnt"))
        self.assertIsNone(command_forbidden("ls /home/user"))


class TestBashPolicy(unittest.TestCase):
    def test_catastrophic_always_refused(self):
        p = BashPolicy(plan_mode=False)
        for cmd in ("rm -rf /", "mkfs.ext4 /dev/sdb", "shutdown now", "dd if=/dev/zero of=/dev/sda"):
            verdict = p.verdict(cmd)
            self.assertIsInstance(verdict, str, cmd)
            self.assertIn("catastrophic", verdict)

    def test_plan_mode_read_only(self):
        p = BashPolicy(plan_mode=True)
        self.assertIsNone(p.verdict("ls /tmp"))
        self.assertIsNone(p.verdict("cat file.txt"))
        self.assertIsInstance(p.verdict("rm file.txt"), str)
        self.assertIsInstance(p.verdict("git add ."), str)
        self.assertIsInstance(p.verdict("echo hi > file.txt"), str)
        self.assertIsInstance(p.verdict("echo $(ls)"), str)

    def test_plan_mode_git_option_flags_blocked(self):
        """git global options (-C, --git-dir, -c ...) must not smuggle a
        mutating subcommand past the plan-mode read-only gate."""
        p = BashPolicy(plan_mode=True)
        for cmd in (
            "git -C /tmp/x push",
            "git --git-dir=/tmp/x push",
            "git -c user.name=x commit",
            "git --work-tree /tmp/x reset --hard",
            "git --namespace=foo rebase main",
        ):
            verdict = p.verdict(cmd)
            self.assertIsInstance(verdict, str, cmd)
        self.assertIsNone(p.verdict("git -C /tmp/x status"))
        self.assertIsNone(p.verdict("git --git-dir=/tmp/x log -1"))

    def test_dangerous_confirm(self):
        p = BashPolicy(approval="confirm", confirm_allowed=True)
        self.assertEqual(p.verdict("git push --force"), "CONFIRM")
        p2 = BashPolicy(approval="confirm", confirm_allowed=False)
        self.assertIsNone(p2.verdict("git push --force"))

    def test_dangerous_block(self):
        p = BashPolicy(approval="block")
        verdict = p.verdict("rm -rf build/")
        self.assertIsInstance(verdict, str)

    def test_destructive_runs_unless_block(self):
        p = BashPolicy(approval="confirm")
        self.assertIsNone(p.verdict("killall node"))
        p2 = BashPolicy(approval="block")
        self.assertIsInstance(p2.verdict("killall node"), str)

    def test_session_allow_deny(self):
        p = BashPolicy(approval="confirm")
        p.session_allow.add("git reset --hard")
        self.assertIsNone(p.verdict("git reset --hard"))
        p.session_deny.add("rm -r x")
        self.assertIn("denied", p.verdict("rm -r x"))

    def test_read_only_bash_helpers(self):
        self.assertTrue(bash_read_only_p("ls -la"))
        self.assertTrue(bash_read_only_p("git status"))
        self.assertFalse(bash_read_only_p("git commit -m x"))
        self.assertFalse(bash_read_only_p("find . -delete"))
        self.assertFalse(bash_read_only_p("sort file -o out"))
        self.assertFalse(bash_read_only_p("ls | rm -rf x"))

    def test_git_option_flags_no_mutating_bypass(self):
        self.assertFalse(bash_read_only_p("git -C /tmp/x push"))
        self.assertFalse(bash_read_only_p("git --git-dir=/tmp/x push"))
        self.assertFalse(bash_read_only_p("git -c user.name=x commit"))
        self.assertFalse(bash_read_only_p("git --work-tree /tmp/x rebase main"))
        self.assertTrue(bash_read_only_p("git -C /tmp/x status"))
        self.assertTrue(bash_read_only_p("git --git-dir=/tmp/x log -1"))


class TestUndo(unittest.TestCase):
    def test_undo_edit(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "f.txt")
            with open(path, "w") as f:
                f.write("original")
            stack = UndoStack(backup_dir=os.path.join(d, "backups"))
            stack.snapshot(path, "Edit")
            with open(path, "w") as f:
                f.write("changed")
            ok, msg = stack.undo_last()
            self.assertTrue(ok)
            with open(path) as f:
                self.assertEqual(f.read(), "original")

    def test_undo_created_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "new.txt")
            stack = UndoStack(backup_dir=os.path.join(d, "backups"))
            stack.snapshot(path, "Write")
            with open(path, "w") as f:
                f.write("x")
            ok, _ = stack.undo_last()
            self.assertTrue(ok)
            self.assertFalse(os.path.exists(path))

    def test_undo_failure_keeps_entry(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "f.txt")
            with open(path, "w") as f:
                f.write("original")
            stack = UndoStack(backup_dir=os.path.join(d, "backups"))
            stack.snapshot(path, "Edit")
            # delete the backup to force failure
            os.remove(stack.entries[-1].backup)
            ok, msg = stack.undo_last()
            self.assertFalse(ok)
            self.assertEqual(len(stack.entries), 0)  # missing backup drops entry

    def test_history(self):
        stack = UndoStack(backup_dir="/tmp/nonexistent-undo-test")
        stack.record_absent("/tmp/nonexistent-undo-test-a.txt", "Write")
        lines = stack.history()
        self.assertEqual(len(lines), 1)
        self.assertIn("Write", lines[0])

    def test_bash_cancel_kills_process(self):
        """A cancelled run must kill the whole process group quickly."""
        import threading
        import time

        from python_agent_harness.tools.base import ToolContext
        from python_agent_harness.tools.bash import Bash

        class Sess:
            def __init__(self):
                self.cancel_event = threading.Event()
                self.bash_timeout = 60

            @property
            def project_dir(self):
                return "/tmp"

        s = Sess()
        s.cancel_event.set()  # cancel is already requested
        ctx = ToolContext(s)
        start = time.monotonic()
        result = Bash().run({"command": "sleep 60"}, ctx)
        elapsed = time.monotonic() - start
        self.assertIn("cancelled", result)
        self.assertLess(elapsed, 5, f"cancel took {elapsed:.1f}s — process group not killed")


if __name__ == "__main__":
    unittest.main()
