"""Edit undo: file snapshots with restore.

Ported from gptel-agent-harness-safety.el: each Edit/Write/Insert snapshots
the file (or records its absence) into a backup dir; undo-last-edit
restores the newest snapshot (or removes a created file).  Entries are
kept on restore failure (retryable); missing backups drop the entry.
"""

from __future__ import annotations

import os
import random
import shutil
import string
import time
from dataclasses import dataclass

from . import config


@dataclass
class UndoEntry:
    path: str
    backup: str | None
    existed: bool
    tool: str
    time: float


class UndoStack:
    def __init__(self, backup_dir: str | None = None, depth: int = config.UNDO_DEPTH) -> None:
        self.depth = depth
        self.backup_dir = backup_dir or os.path.join(
            temp_dir(), "python-agent-harness-undo"
        )
        self.entries: list[UndoEntry] = []

    def snapshot(self, path: str, tool: str) -> None:
        """Snapshot PATH before a write; records absent files separately."""
        path = os.path.abspath(path)
        if os.path.isfile(path):
            os.makedirs(self.backup_dir, exist_ok=True)
            suffix = "".join(random.choices(string.ascii_lowercase, k=6))
            backup = os.path.join(
                self.backup_dir, f"snap-{os.path.basename(path)}-{suffix}"
            )
            try:
                shutil.copy2(path, backup)
            except OSError:
                return
            entry = UndoEntry(path, backup, True, tool, time.time())
        else:
            entry = UndoEntry(path, None, False, tool, time.time())
        self.entries.append(entry)
        if len(self.entries) > self.depth:
            dropped = self.entries.pop(0)
            if dropped.backup and os.path.exists(dropped.backup):
                try:
                    os.remove(dropped.backup)
                except OSError:
                    pass

    def record_absent(self, path: str, tool: str) -> None:
        """Record a file that did not exist before a Write."""
        path = os.path.abspath(path)
        if os.path.exists(path) or any(e.path == path for e in self.entries):
            return
        self.entries.append(UndoEntry(path, None, False, tool, time.time()))

    def undo_last(self) -> tuple[bool, str]:
        """Restore the newest entry. Returns (ok, message)."""
        if not self.entries:
            return False, "Nothing to undo."
        entry = self.entries[-1]
        if entry.existed:
            if entry.backup and os.path.exists(entry.backup):
                try:
                    os.makedirs(os.path.dirname(entry.path), exist_ok=True)
                    shutil.copy2(entry.backup, entry.path)
                except OSError as e:
                    # keep the entry: retryable
                    return False, f"Error: restore failed — {e}"
                self.entries.pop()
                return True, f"Restored {entry.path} (from {entry.tool})"
            # backup missing: drop entry with message
            self.entries.pop()
            return False, f"Backup missing for {entry.path}; entry dropped."
        # file was created by the tool -> remove it
        try:
            if os.path.exists(entry.path):
                os.remove(entry.path)
            self.entries.pop()
            return True, f"Removed created file {entry.path} (from {entry.tool})"
        except OSError as e:
            return False, f"Error: remove failed — {e}"

    def history(self) -> list[str]:
        out = []
        for e in reversed(self.entries):
            stamp = time.strftime("%H:%M:%S", time.localtime(e.time))
            out.append(f"{stamp} {e.tool} {e.path}")
        return out


def temp_dir() -> str:
    import tempfile

    return tempfile.gettempdir()
