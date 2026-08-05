"""Bash tool with layered safety policy.

Tiers (in order, first match wins):
1. forbidden path token in command          -> always refused
2. catastrophic pattern                     -> always refused (before plan gate)
3. plan mode                                -> segment-wise read-only check
4. destructive pattern                      -> run unless approval == block
5. dangerous pattern                        -> verdict (session allow/deny, confirm, block)
6. everything else                          -> run

Timeout: per-call timer; the process is killed and a specific error
message is delivered.  A session cancel (Ctrl-C) also kills the process.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
import time

from .base import Tool, ToolContext


def _split_tokens(command: str) -> list[str]:
    return re.split(r"[ \t\n\r;&|<>()\"']+", command)


class SafetyViolation(Exception):
    pass


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
        "Subject to the harness safety policy (forbidden paths, "
        "catastrophic/destructive/dangerous command patterns, "
        "timeout, plan-mode read-only whitelist)."
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
        verdict = ctx.verify_bash(command)
        if isinstance(verdict, str):
            return verdict
        return self._execute(command, ctx)

    def _execute(self, command: str, ctx: ToolContext) -> str:
        timeout = ctx.bash_timeout
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
        timed_out = threading.Event()
        cancelled = threading.Event()
        cancel = ctx.cancel_event

        def killer() -> None:
            deadline = time.monotonic() + timeout if timeout and timeout > 0 else None
            while True:
                if done.wait(0.05):
                    return
                if cancel is not None and cancel.is_set():
                    cancelled.set()
                    _kill_process(proc)
                    return
                if deadline is not None and time.monotonic() >= deadline:
                    timed_out.set()
                    _kill_process(proc)
                    return

        if timeout and timeout > 0:
            threading.Thread(target=killer, daemon=True).start()

        try:
            out, _ = proc.communicate()
        finally:
            done.set()

        if cancelled.is_set():
            return "Error: Bash command cancelled."
        if timed_out.is_set():
            return (
                f"Error: Bash command timed out after {timeout} seconds "
                f"and was killed.\nCommand:\n{command}"
            )
        return out or ""
