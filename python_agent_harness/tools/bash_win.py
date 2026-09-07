"""Windows Bash tool: process management for win32.

Windows lacks ``os.killpg``, ``os.set_blocking``, ``select.select``
on pipe fds, and ``start_new_session``.  This subclass overrides
``_execute`` to use Windows-native process creation and termination
while inheriting the shared output-truncation, timeout, and exit-code
helpers from :class:`Bash`.

Process creation uses ``CREATE_NEW_PROCESS_GROUP`` so the child is
isolated from console Ctrl-C signals (the TUI handles cancel itself).
Process termination uses ``taskkill /F /T /PID`` to kill the process
and all its children (the Windows equivalent of killing a process
group).  Output is collected via a reader thread + queue, since
``select.select`` cannot poll Windows pipes.

Only the process-creation and output-collection strategy differs;
the tool name, description, parameters, timeout semantics, and
exit-code reporting are inherited so callers, the tool registry,
and the plan-mode write guard are platform-independent.
"""

from __future__ import annotations

import codecs
import contextlib
import os
import queue
import subprocess
import threading
import time
from collections import deque

from ..config import BASH_TIMEOUT_MAX, BASH_TIMEOUT_SILENCE
from ..config import MAX_OUTPUT_CHARS as _MAX_OUTPUT
from .base import PendingToolResult, ToolContext
from .bash import (
    _DRAIN_GRACE,
    _POLL_INTERVAL,
    _READ_CHUNK,
    _TAIL_LINES,
    Bash,
    _append_exit_code,
    _assemble_truncated,
    _timeout_message,
)


def _kill_process_tree(pid: int) -> None:
    """Kill the process PID and all its children (Windows equivalent of
    ``os.killpg`` with ``SIGKILL``).

    Uses ``taskkill /F /T /PID`` which recursively terminates the
    process tree.  Errors are suppressed (already gone, etc.).
    """
    with contextlib.suppress(Exception):
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )


def _kill_process_tree_graceful(pid: int, proc: subprocess.Popen) -> None:
    """Graceful kill: terminate, then taskkill /F /T if still alive.

    Windows has no SIGTERM equivalent for process trees;
    ``proc.terminate()`` calls ``TerminateProcess`` on the immediate
    child.  If it doesn't die within 2s, ``taskkill /F /T`` force-kills
    the whole tree (children included).
    """
    with contextlib.suppress(ProcessLookupError, OSError):
        proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        _kill_process_tree(pid)


def _collect_output_win(proc: subprocess.Popen, cancel: threading.Event | None) -> tuple[str, str]:
    """Read PROC's merged output on Windows; return (text, status).

    Windows lacks ``select.select`` on pipe fds and ``os.set_blocking``,
    so a reader thread does blocking ``os.read`` calls and feeds chunks
    into a ``queue.Queue``.  The collector polls the queue with
    ``_POLL_INTERVAL``, checking cancel/timeout conditions between
    reads — mirroring the Unix ``_collect_output`` logic exactly.

    Status is one of ``"ok"``, ``"cancelled"``, ``"timeout_silence"``,
    ``"timeout_max"``.  Keeps the head (first ``_MAX_OUTPUT`` chars)
    and the tail (last ``_TAIL_LINES`` lines) and discards the middle,
    so memory stays bounded no matter how much the process writes.
    """
    stdout = proc.stdout
    if stdout is None:
        return "", "ok"

    chunk_queue: queue.Queue[bytes | None] = queue.Queue()

    def _reader() -> None:
        """Blocking reader thread: reads raw bytes and feeds the queue."""
        try:
            fd = stdout.fileno()
            while True:
                raw = os.read(fd, _READ_CHUNK)
                if not raw:
                    break
                chunk_queue.put(raw)
        except OSError:
            pass
        finally:
            chunk_queue.put(None)

    threading.Thread(target=_reader, daemon=True).start()

    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    head: list[str] = []
    head_len = 0
    tail: deque[str] = deque(maxlen=_TAIL_LINES)
    pending_line = ""
    total = 0
    exited = False
    drain_until: float | None = None
    start = time.monotonic()
    last_output = start

    def finish() -> str:
        nonlocal pending_line
        if pending_line:
            if len(pending_line) > _MAX_OUTPUT:
                pending_line = pending_line[:_MAX_OUTPUT]
            tail.append(pending_line)
        head_text = "".join(head)
        if total > _MAX_OUTPUT:
            return _assemble_truncated(head_text, tail)
        return head_text

    while True:
        if cancel is not None and cancel.is_set():
            return "", "cancelled"
        now = time.monotonic()
        if not exited:
            if BASH_TIMEOUT_SILENCE is not None and now - last_output >= BASH_TIMEOUT_SILENCE:
                return finish(), "timeout_silence"
            if BASH_TIMEOUT_MAX is not None and now - start >= BASH_TIMEOUT_MAX:
                return finish(), "timeout_max"
        if exited and drain_until is not None and now >= drain_until:
            break
        try:
            raw = chunk_queue.get(timeout=_POLL_INTERVAL)
        except queue.Empty:
            if not exited and proc.poll() is not None:
                exited = True
                drain_until = time.monotonic() + _DRAIN_GRACE
            continue
        if raw is None:
            break
        chunk = decoder.decode(raw)
        total += len(chunk)
        last_output = time.monotonic()
        if head_len < _MAX_OUTPUT:
            take = chunk[: _MAX_OUTPUT - head_len]
            head.append(take)
            head_len += len(take)
        parts = chunk.split("\n")
        parts[0] = pending_line + parts[0]
        pending_line = parts.pop()
        for line in parts:
            if len(line) > _MAX_OUTPUT:
                line = line[:_MAX_OUTPUT]
            tail.append(line)
    return finish(), "ok"


class BashWindows(Bash):
    """Bash for Windows: ``CREATE_NEW_PROCESS_GROUP`` + ``taskkill``.

    Overrides :meth:`_execute` to use Windows-native process creation
    (``CREATE_NEW_PROCESS_GROUP``) and termination (``taskkill /F /T``).
    All shared logic — timeout messages, output truncation, exit-code
    reporting, the ``PendingToolResult`` async contract — is inherited
    unchanged from :class:`Bash`.
    """

    def _execute(self, command: str, ctx: ToolContext) -> str | PendingToolResult:
        cancel = ctx.cancel_event
        if cancel is not None and cancel.is_set():
            return "Error: Bash command cancelled."
        try:
            proc = subprocess.Popen(
                command,
                shell=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                cwd=ctx.cwd,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
        except OSError as e:
            return f"Error: {e}"

        pid = proc.pid
        pending = PendingToolResult()

        def deliverer() -> None:
            try:
                out, status = _collect_output_win(proc, cancel)
            except Exception as e:  # noqa: BLE001 - delivered as an error string
                out = f"Error: Bash failed — {e}"
            else:
                if status == "cancelled":
                    _kill_process_tree(pid)
                    out = "Error: Bash command cancelled."
                elif status == "timeout_silence":
                    _kill_process_tree_graceful(pid, proc)
                    out = _timeout_message(out, silence=True)
                elif status == "timeout_max":
                    _kill_process_tree_graceful(pid, proc)
                    out = _timeout_message(out, silence=False)
                elif status == "ok":
                    out = _append_exit_code(out, proc)
            pending.deliver(out)
            with contextlib.suppress(Exception):
                if proc.stdout is not None:
                    proc.stdout.close()
            with contextlib.suppress(Exception):
                proc.wait(timeout=2)

        threading.Thread(target=deliverer, daemon=True).start()
        return pending
