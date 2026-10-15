"""Relative tool paths resolve against the session's project dir.

Regression: Read/Edit/Insert/Write/Mkdir/Grep/Glob used to resolve a
relative path against the PROCESS working directory, while Bash (and
LSP/diffapply) used ``ToolContext.cwd`` (the project dir).  Under
``--project`` or after a project switch the two differ, so ``Read
f.txt`` and ``Bash cat f.txt`` read different files and Edit/Write could
modify a file outside the project.  The plan-mode guard must resolve
the same way, or it would vet a different file than the tool writes.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from python_agent_harness.session.session import Session
from python_agent_harness.tools import Edit, GlobTool, Grep, Insert, Mkdir, Read, Write
from python_agent_harness.tools.base import ToolContext, resolve_path
from tests.support import session_sandbox  # noqa: F401,E402  (side-effect: redirect SESSION_DIR)


class _ProjectSession:
    """Minimal ToolContext runtime whose project dir is not the cwd."""

    def __init__(self, project_dir: str) -> None:
        self.project_dir = project_dir
        self.cancel_event = threading.Event()

    def __getattr__(self, name: str):  # ask_questions, record_diff, ...
        return lambda *a, **k: None


class TestProjectRelativePaths(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="pah-relpath-")
        self.addCleanup(self._tmp.cleanup)
        root = os.path.realpath(self._tmp.name)
        self.project = os.path.join(root, "project")
        self.elsewhere = os.path.join(root, "elsewhere")
        for d in (self.project, self.elsewhere):
            os.makedirs(d)
            with open(os.path.join(d, "f.txt"), "w") as f:
                f.write(f"{os.path.basename(d)}\n")
        # Process cwd deliberately differs from the project dir.  addCleanup
        # (not tearDown) so the cwd is restored even if setUp fails below.
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(self.elsewhere)
        self.ctx = ToolContext(_ProjectSession(self.project))  # type: ignore[arg-type]

    def _read(self, d: str, name: str = "f.txt") -> str:
        with open(os.path.join(d, name)) as f:
            return f.read()

    def test_resolve_path(self) -> None:
        self.assertEqual(resolve_path("a/b", "/x"), "/x/a/b")
        self.assertEqual(resolve_path("/abs/p", "/x"), "/abs/p")
        self.assertEqual(resolve_path("~", "/x"), os.path.realpath(os.path.expanduser("~")))

    def test_read(self) -> None:
        self.assertEqual(Read().run({"file_path": "f.txt"}, self.ctx), "project\n")

    def test_edit_string_replace(self) -> None:
        out = Edit().run({"path": "f.txt", "old_str": "project", "new_str": "edited"}, self.ctx)
        self.assertNotIn("Error", out)
        self.assertEqual(self._read(self.project), "edited\n")
        self.assertEqual(self._read(self.elsewhere), "elsewhere\n")

    def test_insert(self) -> None:
        out = Insert().run({"path": "f.txt", "line_number": 0, "new_str": "top"}, self.ctx)
        self.assertNotIn("Error", out)
        self.assertEqual(self._read(self.project), "top\nproject\n")
        self.assertEqual(self._read(self.elsewhere), "elsewhere\n")

    def test_write(self) -> None:
        out = Write().run({"path": ".", "filename": "new.txt", "content": "x\n"}, self.ctx)
        self.assertNotIn("Error", out)
        self.assertTrue(os.path.isfile(os.path.join(self.project, "new.txt")))
        self.assertFalse(os.path.exists(os.path.join(self.elsewhere, "new.txt")))

    def test_mkdir(self) -> None:
        Mkdir().run({"parent": ".", "name": "sub"}, self.ctx)
        self.assertTrue(os.path.isdir(os.path.join(self.project, "sub")))
        self.assertFalse(os.path.exists(os.path.join(self.elsewhere, "sub")))

    def test_mkdir_name_is_literal(self) -> None:
        """Only ``parent`` is ~-expanded: a NAME of "~" is a literal dir
        in the project, not a no-op on $HOME."""
        Mkdir().run({"parent": "", "name": "~"}, self.ctx)
        self.assertTrue(os.path.isdir(os.path.join(self.project, "~")))

    def test_grep(self) -> None:
        """Run under every available non-git backend: CI runners may lack
        rg, which once hid that the grep fallback (basic regex) treated
        ``|`` literally and silently matched nothing."""
        real_which = shutil.which
        backends = [b for b in ("rg", "grep") if real_which(b)]
        if not backends:
            self.skipTest("no grep backend")
        for backend in backends:
            hide = {"rg"} if backend == "grep" else set()

            def which(name, *a, _hide=hide, **k):
                return None if name in _hide else real_which(name, *a, **k)

            with self.subTest(backend=backend), mock.patch("shutil.which", side_effect=which):
                out = Grep().run({"regex": "project|elsewhere", "path": "."}, self.ctx)
                self.assertIn(os.path.join(self.project, "f.txt"), out)
                self.assertNotIn(self.elsewhere, out)
                self.assertNotIn("elsewhere", out.replace(self.project, ""))

    def test_glob_explicit_relative_path(self) -> None:
        if not shutil.which("tree"):
            self.skipTest("`tree` not installed (non-git fallback)")
        os.makedirs(os.path.join(self.project, "only_in_project"))
        out = GlobTool().run({"pattern": "*", "path": "only_in_project/.."}, self.ctx)
        self.assertNotIn("not readable", out)
        self.assertIn(self.project, out)

    def test_plan_mode_guard_resolves_like_the_tools(self) -> None:
        stub = mock.Mock(project_dir=self.project)
        expected = os.path.join(self.project, "f.txt")
        for name, args in (
            ("Edit", {"path": "f.txt"}),
            ("Insert", {"path": "f.txt"}),
            ("Write", {"path": ".", "filename": "f.txt"}),
            ("Write", {"path": "f.txt", "filename": ""}),
            ("Mkdir", {"parent": ".", "name": "f.txt"}),
        ):
            with self.subTest(name=name, args=args):
                self.assertEqual(Session._tool_path(stub, name, args), expected)
        self.assertEqual(
            Session._tool_path(stub, "Mkdir", {"parent": "", "name": "~"}),
            os.path.join(self.project, "~"),
        )


class TestBashReaderCrashKillsGroup(unittest.TestCase):
    def test_reader_exception_kills_process_group(self) -> None:
        """If the output reader raises, nothing can cancel or time out
        the command any more: the process group must be killed."""
        from python_agent_harness.tools import bash as bash_mod

        with tempfile.TemporaryDirectory(prefix="pah-crash-") as d:
            pidfile = os.path.join(d, "pid")
            real_collect = bash_mod._collect_output

            def crashing_collect(proc, cancel):
                deadline = time.monotonic() + 3
                while not os.path.exists(pidfile) and time.monotonic() < deadline:
                    time.sleep(0.02)
                raise ValueError("filedescriptor out of range in select()")

            with mock.patch.object(bash_mod, "_collect_output", crashing_collect):
                result = bash_mod.Bash().run(
                    {"command": f"echo $$ > {pidfile}; exec sleep 30"},
                    ToolContext(_ProjectSession(d)),  # type: ignore[arg-type]
                )
                out = result.wait()  # type: ignore[union-attr]
            self.assertIs(bash_mod._collect_output, real_collect)
            self.assertIn("Bash failed", out)
            with open(pidfile) as f:
                pgid = int(f.read().strip())
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                try:
                    os.kill(pgid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.05)
            else:
                os.killpg(pgid, 9)
                self.fail("process group survived a reader crash")


if __name__ == "__main__":
    unittest.main()
