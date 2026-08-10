"""Bash tool.

A session cancel (Ctrl-C) kills the process.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading

from .base import Tool, ToolContext


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

    def run(self, args: dict, ctx: ToolContext) -> str:
        command = args["command"]
        return self._execute(command, ctx)

    def _execute(self, command: str, ctx: ToolContext) -> str:
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

        done = threading.Event()
        cancelled = threading.Event()
        cancel = ctx.cancel_event

        def watcher() -> None:
            while True:
                if done.wait(0.05):
                    return
                if cancel is not None and cancel.is_set():
                    cancelled.set()
                    _kill_process(proc)
                    return

        threading.Thread(target=watcher, daemon=True).start()

        try:
            out, _ = proc.communicate()
        finally:
            done.set()

        if cancelled.is_set():
            return "Error: Bash command cancelled."
        return out or ""
