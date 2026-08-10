"""Bash tool.

Asynchronous (mirrors ``:async t`` in gptel-agent-tools): ``run``
spawns the process and returns a ``PendingToolResult`` immediately; a
background thread collects the output and delivers it when the process
exits.  A long-running command therefore never occupies a thread-pool
slot — sibling tools keep their slots and the round completes by
delivery, not by thread blocking.

A session cancel (Ctrl-C) kills the process group and delivers a
cancelled error.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading

from .base import PendingToolResult, Tool, ToolContext


def _kill_process(proc: subprocess.Popen) -> None:
    """Kill PROC and its whole process group.

    The process is started in its own session so a shell's children
    (e.g. `sleep 30` spawned by the shell) die too; otherwise they keep
    the stdout pipe open and ``communicate()`` blocks until they exit.
    """
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass


class Bash(Tool):
    name = "Bash"
    description = (
        "Execute a shell command. Returns stdout, or an error string. "
        "A session cancel (Ctrl-C) kills the process."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The shell command to run"},
        },
        "required": ["command"],
    }

    def run(self, args: dict, ctx: ToolContext) -> str | PendingToolResult:
        command = args["command"]
        return self._execute(command, ctx)

    def _execute(self, command: str, ctx: ToolContext) -> str | PendingToolResult:
        try:
            proc = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=ctx.cwd,
                start_new_session=True,
            )
        except OSError as e:
            return f"Error: {e}"

        pending = PendingToolResult()
        cancel = ctx.cancel_event
        done = threading.Event()
        killed = threading.Event()

        def watcher() -> None:
            """Kill the process group when the session is cancelled."""
            while True:
                if done.wait(0.05):
                    return
                if cancel is not None and cancel.is_set():
                    killed.set()
                    _kill_process(proc)
                    return

        def deliverer() -> None:
            """Collect output; deliver it once the process exits."""
            try:
                out, _ = proc.communicate()
            except Exception as e:  # noqa: BLE001 - delivered as an error string
                out = f"Error: Bash failed — {e}"
            finally:
                done.set()
            if killed.is_set():
                pending.deliver("Error: Bash command cancelled.")
            else:
                pending.deliver(out or "")

        threading.Thread(target=watcher, daemon=True).start()
        threading.Thread(target=deliverer, daemon=True).start()
        return pending
