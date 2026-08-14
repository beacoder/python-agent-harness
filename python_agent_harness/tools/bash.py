"""Bash tool.

Asynchronous (mirrors ``:async t`` in gptel-agent-tools): ``run``
spawns the process and returns a ``PendingToolResult`` immediately; a
background thread collects the output and delivers it when the process
exits.  A long-running command therefore never blocks the parent's
sequential tool loop — it runs concurrently with sibling async tools
(Agent) while sync tools execute one at a time.

A session cancel (Ctrl-C) kills the process group and delivers a
cancelled error.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading

from .base import PendingToolResult, Tool, ToolContext

# Maximum output size (chars) before truncation.  Matches the 20 KB
# spool threshold used by Glob/Grep/Read so that a model-issued
# command cannot exhaust memory.
_MAX_OUTPUT = 20_000
_TAIL_LINES = 50  # lines kept from the tail after truncation


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
                stdin=subprocess.DEVNULL,
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
                out = out or ""
                if len(out) > _MAX_OUTPUT:
                    # Keep head + tail so the model sees the start and end
                    tail = "\n".join(out.splitlines()[-_TAIL_LINES:])
                    head_budget = _MAX_OUTPUT - len(tail) - 200  # room for notice
                    head = out[:max(head_budget, 1000)]
                    out = (
                        f"{head}\n\n... [truncated: output exceeded "
                        f"{_MAX_OUTPUT} chars] ...\n\n{tail}"
                    )
                pending.deliver(out)

        threading.Thread(target=watcher, daemon=True).start()
        threading.Thread(target=deliverer, daemon=True).start()
        return pending
