"""Filesystem tool helpers and compatibility re-exports.

This module hosts the SHARED helper machinery for the filesystem
tools — spooling oversized tool results to temp files
(`gptel-agent--truncate-buffer` parity), git-root detection, and the
``natnump`` predicate — and re-exports the tool classes that live in
per-tool modules (`read.py`, `glob.py`, `grep.py`, `edit.py`,
`write.py`, `insert.py`, `mkdir.py`), so existing imports such as
``from .tools.filesystem import Read`` keep working.

The helpers must stay defined HERE (not in a separate ``_common``
module): tests monkey-patch ``filesystem._spool_dir`` and read
``filesystem.MAX_OUTPUT`` / ``SPOOL_LINES`` / ``READ_SIZE_LIMIT`` off
this module's namespace, and the helper functions resolve them through
their defining module's globals.

NOTE: these filesystem tools are intentionally SYNCHRONOUS.  Only Bash
and Agent are `:async t` in gptel-agent; sync tools run one at a time in
the model-emitted order.  Do NOT be tempted to port every tool to async
for parallelism: tools can depend on one another's side effects within a
single round (e.g. Write/Mkdir then Read/Edit the same path, or Edit then
Grep the just-changed file).  Running them concurrently would introduce
read-after-write races and non-deterministic results.  Keep filesystem
tools synchronous so ordering — and therefore correctness — is preserved.

Oversized Glob/Grep results are spilled to a temp file (mirroring
`gptel-agent--truncate-buffer` in gptel-agent-tools.el): the tool
result then carries a short preview plus the temp-file path, so the
full output remains readable via the Read tool.
"""

from __future__ import annotations

import os
import shutil  # noqa: F401  (mock target for tests)
import subprocess  # noqa: F401  (mock target for tests)
import tempfile
import threading
import time
from pathlib import Path
from typing import TypeGuard

from ..config import MAX_OUTPUT_CHARS as MAX_OUTPUT

SPOOL_LINES = 50  # preview lines kept when results are spilled
READ_SIZE_LIMIT = 400 * 1024  # whole-file reads above this are refused
# (mirrors gptel-agent-read-file-size-threshold)

_spooled_files: list[str] = []  # temp files created by _spool, cleaned
# up by cleanup_spooled_files on session close
_spooled_files_lock = threading.Lock()  # guards _spooled_files across
# parallel readonly tool threads (Glob/Grep/Read can _spool concurrently)


def _truncate(text: str, label: str = "output") -> str:
    """In-memory truncation fallback (used when spooling to disk fails)."""
    if len(text) > MAX_OUTPUT:
        return text[:MAX_OUTPUT] + f"\n... [truncated {label}]"
    return text


def _spool_dir() -> str:
    """Reliable temp dir for spilled results (first candidate set, else /tmp)."""
    for d in (
        os.environ.get("TMPDIR"),
        os.environ.get("TMP"),
        os.environ.get("TEMP"),
        tempfile.gettempdir(),
    ):
        if d:
            return os.path.abspath(d)
    return "/tmp"


def _spool(text: str, label: str) -> str:
    """Spill oversized tool output to a temp file; return a preview.

    Mirrors `gptel-agent--truncate-buffer': when TEXT exceeds
    MAX_OUTPUT chars the full content is written to a temp file and the
    returned string becomes a header (size + path), the first
    SPOOL_LINES lines, and a footer telling the agent to Read the file.
    Falls back to in-memory truncation if the temp file cannot be
    written."""
    if len(text) <= MAX_OUTPUT:
        return text
    stamp = time.strftime("%Y%m%d-%H%M%S")
    try:
        fd, temp_file = tempfile.mkstemp(
            prefix=f"python-agent-harness-{label}-{stamp}-",
            suffix=".txt",
            dir=_spool_dir(),
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        with _spooled_files_lock:
            _spooled_files.append(temp_file)
    except OSError:
        return _truncate(text, label)
    lines = text.splitlines()
    preview = "\n".join(lines[:SPOOL_LINES])
    return (
        f"{label} results too large ({len(text)} chars, {len(lines)} lines) "
        f"for context window.\n"
        f"Stored in: {temp_file}\n\n"
        f"First {SPOOL_LINES} lines:\n\n"
        f"{preview}\n\n"
        f'[Use Read tool with file_path="{temp_file}" to view full results]'
    )


def cleanup_spooled_files() -> None:
    """Delete all tracked spooled temp files (best effort).

    Mirrors ``PlanMode.cleanup_plan_file``: called from
    ``Session.close`` so oversized tool results do not accumulate
    in the temp dir.  Files already removed (e.g. by a restored
    session) are skipped.
    """
    with _spooled_files_lock:
        paths = _spooled_files[:]
        _spooled_files.clear()
    for path in paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass


def _natnump(n: object) -> TypeGuard[int]:
    """True for a non-negative integer (Emacs `natnump' semantics)."""
    return isinstance(n, int) and not isinstance(n, bool) and n >= 0


def _git_root(path: str) -> str | None:
    d = Path(path).resolve()
    for parent in [d, *d.parents]:
        # .git is a directory in a normal clone and a file in worktrees
        # / submodules; exists() covers both.
        if (parent / ".git").exists():
            return str(parent)
    return None


# ---------------------------------------------------------------------------
# tool classes live in per-tool modules; re-exported here so existing
# `from .tools.filesystem import ...` imports (tests included) keep working.
# The imports sit at the BOTTOM on purpose: the per-tool modules import the
# shared helpers from this module, so the helpers above must be defined
# before those modules are loaded (this is the fixed import order — do not
# move these imports to the top of the file).
# ---------------------------------------------------------------------------
# isort: off
from .edit import Edit, _fix_patch_headers, _strip_diff_fence  # noqa: E402
from .glob import GlobTool, _git_glob_results  # noqa: E402
from .glob_mac import GlobMac  # noqa: E402
from .grep import Grep, _grep_out  # noqa: E402
from .grep_mac import GrepMac  # noqa: E402
from .insert import Insert  # noqa: E402
from .mkdir import Mkdir  # noqa: E402
from .read import Read  # noqa: E402
from .write import Write  # noqa: E402
# isort: on

__all__ = [
    "Edit",
    "GlobMac",
    "GlobTool",
    "Grep",
    "GrepMac",
    "Insert",
    "Mkdir",
    "Read",
    "Write",
    "MAX_OUTPUT",
    "READ_SIZE_LIMIT",
    "SPOOL_LINES",
    "cleanup_spooled_files",
    "_fix_patch_headers",
    "_git_glob_results",
    "_git_root",
    "_grep_out",
    "_natnump",
    "_spool",
    "_spool_dir",
    "_spooled_files",
    "_spooled_files_lock",
    "_strip_diff_fence",
    "_truncate",
]
