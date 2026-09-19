"""Tests for the Windows Bash tool (bash_win.py).

``BashWindows`` normally runs on win32, but its process-management and
output-collection strategy is exercisable everywhere:

- ``creationflags`` is resolved via ``getattr(subprocess,
  "CREATE_NEW_PROCESS_GROUP", 0)``, so ``Popen`` works unchanged on
  POSIX (flag 0) — real subprocess tests run on every CI OS.
- ``taskkill`` termination is mocked (it does not exist off-Windows).
- The reader-thread + queue collector is process-agnostic; real pipes
  feed it on any OS.

Commands are built from ``sys.executable`` so they run under any
shell (no POSIX-only ``sleep``/``echo`` assumptions).
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from python_agent_harness.tools import bash_win
from python_agent_harness.tools.base import PendingToolResult, ToolContext
from python_agent_harness.tools.bash_win import (
    BashWindows,
    _collect_output_win,
    _kill_process_tree,
    _kill_process_tree_graceful,
)


class FakeSession:
    """Minimal runtime double: bash tools only read cwd + cancel_event."""

    def __init__(self) -> None:
        # NOT "/tmp": every BashWindows run passes cwd=ctx.cwd to Popen,
        # and a missing cwd raises OSError on win32 (no "/tmp" there),
        # silently turning real subprocess tests into error-string paths.
        self.project_dir = tempfile.gettempdir()
        self._cancel = threading.Event()

    @property
    def cancel_event(self) -> threading.Event:
        return self._cancel


def _py(code: str) -> str:
    """A shell command running CODE under the current interpreter."""
    return f'"{sys.executable}" -c "{code}"'


def _spawn(command: str) -> subprocess.Popen:
    return subprocess.Popen(
        command,
        shell=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )


class TestKillProcessTree(unittest.TestCase):
    def test_taskkill_invoked_with_force_and_tree(self):
        with mock.patch.object(bash_win.subprocess, "run") as run:
            _kill_process_tree(4242)
        run.assert_called_once()
        args, kwargs = run.call_args
        self.assertEqual(args[0], ["taskkill", "/F", "/T", "/PID", "4242"])
        self.assertEqual(kwargs["timeout"], 10)

    def test_errors_are_suppressed(self):
        with mock.patch.object(
            bash_win.subprocess, "run", side_effect=FileNotFoundError("no taskkill")
        ):
            _kill_process_tree(1)  # must not raise


class TestKillProcessTreeGraceful(unittest.TestCase):
    def test_terminate_then_no_taskkill_when_it_dies(self):
        proc = mock.Mock()
        with mock.patch.object(bash_win, "_kill_process_tree") as kill:
            _kill_process_tree_graceful(7, proc)
        proc.terminate.assert_called_once()
        proc.wait.assert_called_once_with(timeout=2)
        kill.assert_not_called()

    def test_taskkill_fallback_when_terminate_times_out(self):
        proc = mock.Mock()
        proc.wait.side_effect = subprocess.TimeoutExpired("x", 2)
        with mock.patch.object(bash_win, "_kill_process_tree") as kill:
            _kill_process_tree_graceful(9, proc)
        kill.assert_called_once_with(9)

    def test_already_dead_process_is_tolerated(self):
        proc = mock.Mock()
        proc.terminate.side_effect = ProcessLookupError
        with mock.patch.object(bash_win, "_kill_process_tree") as kill:
            _kill_process_tree_graceful(3, proc)
        kill.assert_not_called()  # wait() succeeded; nothing to force-kill


class TestCollectOutputWin(unittest.TestCase):
    def test_stdout_none_yields_empty_ok(self):
        proc = mock.Mock()
        proc.stdout = None
        self.assertEqual(_collect_output_win(proc, None), ("", "ok"))

    def test_collects_merged_output(self):
        proc = _spawn(_py("print('hello')"))
        try:
            out, status = _collect_output_win(proc, None)
        finally:
            proc.wait(timeout=5)
        self.assertEqual(status, "ok")
        self.assertEqual(out, "hello\n")

    def test_crlf_normalized_and_lines_kept(self):
        """Explicit CRLF bytes normalize to LF.  Written through
        ``sys.stdout.buffer`` (binary) so Windows' text-mode newline
        translation cannot turn each ``\\n`` into a second CR — in text
        mode the emit would be ``\\r\\r\\n`` and the collector's
        ``\\r\\n`` -> ``\\n`` pass would leave a stray ``\\r`` that
        fails the exact-match assertion."""
        proc = _spawn(
            _py("import sys; sys.stdout.buffer.write(b'a\\r\\nb\\r\\n'); sys.stdout.buffer.flush()")
        )
        try:
            out, status = _collect_output_win(proc, None)
        finally:
            proc.wait(timeout=5)
        self.assertEqual(status, "ok")
        self.assertEqual(out, "a\nb\n")

    def test_reader_oserror_is_contained(self):
        """An os.read failure in the reader thread must surface as empty
        output, not a stuck collector (the reader still sends the None
        sentinel from its finally block)."""
        fake = mock.Mock()
        fake.stdout = mock.Mock()
        fake.stdout.fileno.side_effect = OSError("pipe broke")
        out, status = _collect_output_win(fake, None)
        self.assertEqual((out, status), ("", "ok"))

    def test_unterminated_final_line_is_capped(self):
        """A final line with no trailing newline must be length-capped
        when flushed into the tail at finish()."""
        with mock.patch.object(bash_win, "_MAX_OUTPUT", 40):
            proc = _spawn(_py("import sys; sys.stdout.write('z' * 500)"))
            try:
                out, status = _collect_output_win(proc, None)
            finally:
                proc.wait(timeout=5)
        self.assertEqual(status, "ok")
        self.assertIn("[truncated", out)
        self.assertLess(len(out), 600)

    def test_detached_grandchild_holding_pipe_does_not_wedge(self):
        """The shell exits while a grandchild keeps the stdout pipe open:
        the collector must stop draining shortly after the exit instead
        of waiting for EOF that never comes."""
        code = (
            "import subprocess, sys; "
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
            "print('done')"
        )
        proc = _spawn(_py(code))
        start = time.monotonic()
        try:
            out, status = _collect_output_win(proc, None)
        finally:
            proc.wait(timeout=5)
        elapsed = time.monotonic() - start
        self.assertEqual(status, "ok")
        self.assertEqual(out, "done\n")
        self.assertLess(elapsed, 5)

    def test_large_output_truncated_head_and_tail(self):
        with mock.patch.object(bash_win, "_MAX_OUTPUT", 60):
            proc = _spawn(_py("print('x' * 30); [print(i) for i in range(50)]"))
            try:
                out, status = _collect_output_win(proc, None)
            finally:
                proc.wait(timeout=5)
        self.assertEqual(status, "ok")
        self.assertIn("[truncated", out)
        self.assertTrue(out.startswith("x" * 30))
        self.assertIn("49", out)  # tail kept

    def test_single_giant_line_is_capped(self):
        with mock.patch.object(bash_win, "_MAX_OUTPUT", 40):
            proc = _spawn(_py("print('y' * 5000)"))
            try:
                out, status = _collect_output_win(proc, None)
            finally:
                proc.wait(timeout=5)
        self.assertEqual(status, "ok")
        self.assertIn("[truncated", out)
        self.assertLess(len(out), 600)

    def test_preexisting_cancel_returns_cancelled(self):
        cancel = threading.Event()
        cancel.set()
        proc = _spawn(_py("import time; time.sleep(5)"))
        try:
            self.assertEqual(_collect_output_win(proc, cancel), ("", "cancelled"))
        finally:
            proc.kill()
            proc.wait(timeout=5)

    def test_silence_timeout_fires_while_quiet(self):
        with mock.patch.object(bash_win, "BASH_TIMEOUT_SILENCE", 0.2):
            proc = _spawn(_py("import time; time.sleep(5)"))
            try:
                _, status = _collect_output_win(proc, None)
            finally:
                proc.kill()
                proc.wait(timeout=5)
        self.assertEqual(status, "timeout_silence")

    def test_max_timeout_fires_despite_output(self):
        with (
            mock.patch.object(bash_win, "BASH_TIMEOUT_SILENCE", None),
            mock.patch.object(bash_win, "BASH_TIMEOUT_MAX", 0.3),
        ):
            proc = _spawn(
                _py("import time; [print(i, flush=True) or time.sleep(0.1) for i in range(300)]")
            )
            try:
                _, status = _collect_output_win(proc, None)
            finally:
                proc.kill()
                proc.wait(timeout=5)
        self.assertEqual(status, "timeout_max")

    def test_activity_resets_silence_timer(self):
        """Activity must keep resetting the silence timer: with a 2.0s
        budget — generous margin for slow process startup on loaded CI
        boxes (Windows runners can spend ~1s before the child's first
        output) — and output every 0.05s for ~3s, the run must complete
        without a timeout.  A broken reset would fire at start+2.0s and
        drop the tail of the output, failing the "59" assertion."""
        with mock.patch.object(bash_win, "BASH_TIMEOUT_SILENCE", 2.0):
            proc = _spawn(
                _py("import time; [print(i, flush=True) or time.sleep(0.05) for i in range(60)]")
            )
            try:
                out, status = _collect_output_win(proc, None)
            finally:
                proc.wait(timeout=5)
        self.assertEqual(status, "ok")
        self.assertIn("59", out)  # ran to completion, no timeout


class TestBashWindowsExecute(unittest.TestCase):
    def setUp(self) -> None:
        self.sess = FakeSession()
        self.ctx = ToolContext(self.sess)

    def test_cancel_preset_skips_spawn(self):
        self.sess._cancel.set()
        result = BashWindows().run({"command": "echo hi"}, self.ctx)
        self.assertIsInstance(result, str)
        self.assertEqual(result, "Error: Bash command cancelled.")

    def test_spawn_oserror_reported(self):
        with mock.patch.object(bash_win.subprocess, "Popen", side_effect=OSError("cannot spawn")):
            result = BashWindows().run({"command": "echo hi"}, self.ctx)
        self.assertIsInstance(result, str)
        self.assertIn("cannot spawn", result)

    def test_success_delivers_output_and_exit_code(self):
        result = BashWindows().run({"command": _py("print(42)")}, self.ctx)
        self.assertIsInstance(result, PendingToolResult)
        out = result.wait()
        self.assertIn("42", out)
        self.assertTrue(out.endswith("Exit code: 0"))

    def test_collector_failure_delivered_as_error(self):
        with mock.patch.object(
            bash_win, "_collect_output_win", side_effect=RuntimeError("pipe boom")
        ):
            result = BashWindows().run({"command": _py("print(1)")}, self.ctx)
        self.assertIn("Error: Bash failed — pipe boom", result.wait())

    def test_cancel_kills_tree_and_reports(self):
        real_popen = subprocess.Popen
        procs: list[subprocess.Popen] = []

        def tracking_popen(*args, **kwargs):
            proc = real_popen(*args, **kwargs)
            procs.append(proc)
            return proc

        with (
            mock.patch.object(bash_win.subprocess, "Popen", side_effect=tracking_popen),
            mock.patch.object(bash_win, "_kill_process_tree") as kill,
        ):
            result = BashWindows().run({"command": _py("import time; time.sleep(30)")}, self.ctx)
            time.sleep(0.3)  # let the child start and go quiet
            self.sess._cancel.set()
            out = result.wait()
        self.assertEqual(out, "Error: Bash command cancelled.")
        kill.assert_called_once_with(procs[0].pid)
        procs[0].kill()
        procs[0].wait(timeout=5)

    def test_silence_timeout_reported_and_gracefully_killed(self):
        real_popen = subprocess.Popen
        procs: list[subprocess.Popen] = []

        def tracking_popen(*args, **kwargs):
            proc = real_popen(*args, **kwargs)
            procs.append(proc)
            return proc

        with (
            mock.patch.object(bash_win.subprocess, "Popen", side_effect=tracking_popen),
            mock.patch.object(bash_win, "BASH_TIMEOUT_SILENCE", 0.3),
            mock.patch.object(bash_win, "_kill_process_tree_graceful") as graceful,
        ):
            result = BashWindows().run({"command": _py("import time; time.sleep(30)")}, self.ctx)
            try:
                out = result.wait()
            finally:
                procs[0].kill()
                procs[0].wait(timeout=5)
        self.assertIn("timed out", out)
        self.assertIn("no output for", out)
        graceful.assert_called_once()

    def test_max_timeout_reported(self):
        real_popen = subprocess.Popen
        procs: list[subprocess.Popen] = []

        def tracking_popen(*args, **kwargs):
            proc = real_popen(*args, **kwargs)
            procs.append(proc)
            return proc

        with (
            mock.patch.object(bash_win.subprocess, "Popen", side_effect=tracking_popen),
            mock.patch.object(bash_win, "BASH_TIMEOUT_SILENCE", None),
            mock.patch.object(bash_win, "BASH_TIMEOUT_MAX", 0.3),
            mock.patch.object(bash_win, "_kill_process_tree_graceful") as graceful,
        ):
            result = BashWindows().run(
                {
                    "command": _py(
                        "import time; [print(i, flush=True) or time.sleep(0.1) for i in range(300)]"
                    )
                },
                self.ctx,
            )
            try:
                out = result.wait()
            finally:
                procs[0].kill()
                procs[0].wait(timeout=5)
        self.assertIn("timed out", out)
        self.assertIn("maximum", out)
        graceful.assert_called_once()

    def test_nonzero_exit_code_reported(self):
        result = BashWindows().run({"command": _py("import sys; sys.exit(3)")}, self.ctx)
        self.assertTrue(result.wait().endswith("Exit code: 3"))


if __name__ == "__main__":
    unittest.main()
