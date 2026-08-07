"""Tool-result cache with deduplication.

Ported from gptel-agent-harness-cache.el.

- Key = (tool-name, args); paths are canonicalized by the caller.
- Validity: file mtime equality for regular files; TTL for directories.
- Write-through invalidation on Edit/Write/Insert: exact path OR
  directory-prefix match against any string argument.
- Dedup per epoch: first hit returns the full result and marks the key
  seen; repeats return a short "[Cached: ...]" marker.  Epoch reset
  clears only the seen set — the table survives compaction.
- Max 200 entries; evict oldest by timestamp.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from . import config

DEDUP_TEMPLATES = {
    "read": '[Cached: Read "%s" (%d chars) — same as earlier call, see above]',
    "read-range": '[Cached: Read "%s" lines %s-%s (%d chars) — same as earlier call, see above]',
    "glob": '[Cached: Glob "%s" in %s (%d chars) — same as earlier call, see above]',
    "grep": '[Cached: Grep "%s" in %s (%d chars) — same as earlier call, see above]',
}


@dataclass
class Entry:
    result: str
    mtime: float | None
    timestamp: float = field(default_factory=time.time)


class ToolCache:
    def __init__(
        self,
        enabled: bool = config.CACHE_ENABLED,
        ttl: int = config.CACHE_TTL,
        max_entries: int = config.CACHE_MAX_ENTRIES,
    ) -> None:
        self.enabled = enabled
        self.ttl = ttl
        self.max_entries = max_entries
        self.table: dict[tuple, Entry] = {}
        self.seen: set[tuple] = set()
        self.hits = 0
        self.misses = 0
        self.dedups = 0
        self.invalidations = 0

    # -- key handling ------------------------------------------------------
    def _file_mtime(self, path: str) -> float | None:
        try:
            return os.path.getmtime(path)
        except OSError:
            return None

    def valid_p(self, entry: Entry, path: str | None) -> bool:
        """Check whether ENTRY is still valid for PATH (if given)."""
        if path and os.path.isfile(path):
            current = self._file_mtime(path)
            return entry.mtime is not None and current is not None and entry.mtime == current
        if path and entry.mtime is not None and not os.path.exists(path):
            return False
        return (time.time() - entry.timestamp) < self.ttl

    # -- core ops ------------------------------------------------------------
    def lookup(self, key: tuple, path: str | None = None) -> Entry | None:
        entry = self.table.get(key)
        if entry is None:
            return None
        if not self.valid_p(entry, path):
            del self.table[key]
            self.seen.discard(key)
            self.invalidations += 1
            return None
        return entry

    def store(self, key: tuple, result: str, path: str | None = None) -> None:
        if len(self.table) >= self.max_entries:
            self._evict_oldest()
        mtime = self._file_mtime(path) if path and os.path.isfile(path) else None
        self.table[key] = Entry(result=result, mtime=mtime)

    def get(self, key: tuple, path: str | None = None) -> str | None:
        """Return the result to deliver: full result, dedup marker, or None.

        - cache miss -> None (caller executes the tool)
        - hit, key not seen -> full result; key marked seen
        - hit, key seen -> dedup marker string
        """
        entry = self.lookup(key, path)
        if entry is None:
            return None
        if key in self.seen:
            self.dedups += 1
            return self.dedup_message(key[0], key[1:], entry.result)
        self.seen.add(key)
        self.hits += 1
        return entry.result

    def mark_seen(self, key: tuple) -> None:
        """Mark KEY as delivered (called after a miss is executed/stored)."""
        if key in self.table:
            self.seen.add(key)

    def _evict_oldest(self) -> None:
        if not self.table:
            return
        oldest_key = min(self.table, key=lambda k: self.table[k].timestamp)
        del self.table[oldest_key]
        self.seen.discard(oldest_key)

    def reset_epoch(self) -> None:
        """Clear only the seen set (compaction boundary)."""
        self.seen.clear()

    def clear(self) -> None:
        self.table.clear()
        self.seen.clear()

    # -- dedup messages -------------------------------------------------------
    def dedup_message(self, tool: str, args: tuple, result: str) -> str:
        """Build the dedup marker for a repeated call."""
        n = len(result)
        if tool == "read":
            if len(args) >= 3 and (args[1] is not None or args[2] is not None):
                return DEDUP_TEMPLATES["read-range"] % (
                    _abbrev(args[0]), args[1], args[2], n
                )
            return DEDUP_TEMPLATES["read"] % (_abbrev(args[0]), n)
        if tool == "glob":
            pattern, base, depth = (args + (None, None, None))[:3]
            return DEDUP_TEMPLATES["glob"] % (pattern, _abbrev(base), n)
        if tool == "grep":
            regex, path, glob, ctx = (args + (None, None, None, None))[:4]
            return DEDUP_TEMPLATES["grep"] % (regex, _abbrev(path), n)
        return f"[Cached: {tool} — same as earlier call, see above]"

    # -- write-through invalidation --------------------------------------------
    def invalidate_path(self, path: str) -> None:
        """Invalidate entries whose key references PATH (exact or dir prefix)."""
        expanded = os.path.abspath(path)
        to_remove = []
        for key in self.table:
            for arg in key[1:]:
                if not isinstance(arg, str) or not arg:
                    continue
                if arg == expanded:
                    to_remove.append(key)
                    break
                try:
                    if expanded.startswith(arg + os.sep):
                        to_remove.append(key)
                        break
                except TypeError:
                    continue
        for key in to_remove:
            del self.table[key]
            self.seen.discard(key)
            self.invalidations += 1

    # -- cacheability ----------------------------------------------------------
    @staticmethod
    def cacheable_p(result: str) -> bool:
        if not isinstance(result, str) or not result:
            return False
        if result.startswith("Error:"):
            return False
        if "failed with exit code" in result:
            return False
        return True


def _abbrev(path: str) -> str:
    home = os.path.expanduser("~")
    if path.startswith(home):
        return "~" + path[len(home):]
    return path
