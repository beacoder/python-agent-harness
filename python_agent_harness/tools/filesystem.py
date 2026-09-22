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
import re
import shutil  # noqa: F401  (mock target for tests)
import subprocess  # noqa: F401  (mock target for tests)
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TypeGuard

from ..session.config import MAX_OUTPUT_CHARS as MAX_OUTPUT

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


def _glob_to_regex(pattern: str) -> str:
    """Translate a pathlib-style glob pattern into a :mod:`re` pattern.

    Follows :mod:`fnmatch` conventions with two deliberate deviations:
    ``*`` does not cross path separators (unlike :func:`fnmatch.fnmatch`)
    and ``**`` matches any number of directories (including none).  A
    pattern without any glob metacharacters is escaped, so plain names
    match exactly.
    """
    i, n = 0, len(pattern)
    out: list[str] = []
    while i < n:
        c = pattern[i]
        if c == "*":
            if pattern[i : i + 2] == "**":
                i += 2
                if pattern[i : i + 1] == "/":
                    i += 1
                out.append("(?:.*/)?")
                continue
            out.append("[^/]*")
            i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        elif c == "[":
            end = _find_glob_class_end(pattern, i)
            if end is None:
                out.append(re.escape(c))
                i += 1
            else:
                # The class body is regex-compatible as-is (\], \\, ranges,
                # and member brackets mean the same thing in both dialects);
                # only the negation sigil differs (! -> ^).
                body = pattern[i + 1 : end]
                if body.startswith("!"):
                    body = "^" + body[1:]
                out.append("[" + body + "]")
                i = end + 1
        else:
            out.append(re.escape(c))
            i += 1
    return "".join(out)


def _find_glob_class_end(pattern: str, start: int) -> int | None:
    """Index of the ``]`` closing the char class opened at ``start``.

    Mirrors glob semantics: a ``]`` immediately after ``[`` (or ``[!``)
    is a literal bracket, and backslash escapes are honoured.
    """
    i = start + 1
    if i < len(pattern) and pattern[i] == "!":
        i += 1
    if i < len(pattern) and pattern[i] == "]":
        i += 1
    while i < len(pattern):
        if pattern[i] == "\\" and i + 1 < len(pattern):
            i += 2
        elif pattern[i] == "]":
            return i
        else:
            i += 1
    return None


def _walk_files(root: Path, onerror: Callable[[OSError], object] | None = None) -> Iterator[Path]:
    """Depth-first file iterator that never raises on traversal errors.

    Wraps :func:`os.walk` with ``onerror`` so unreadable directories are
    reported instead of silently swallowed, and wraps the whole iteration
    in a try/except so a scan that dies mid-flight still yields the
    entries collected so far (mirroring the 3.13+ ``pathlib`` scan
    behaviour on every supported version).
    """
    try:
        for dirpath, _dirnames, filenames in os.walk(root, onerror=onerror, followlinks=False):
            for name in filenames:
                yield Path(dirpath) / name
    except OSError as e:
        if onerror is not None:
            onerror(e)


def _git_root(path: str) -> str | None:
    d = Path(path).resolve()
    for parent in [d, *d.parents]:
        # .git is a directory in a normal clone and a file in worktrees
        # / submodules; exists() covers both.
        if (parent / ".git").exists():
            return str(parent)
    return None


def atomic_write_text(path: str, content: str) -> None:
    """Replace PATH's contents with CONTENT atomically.

    THE single write path for every tool that rewrites a file (``Edit``,
    ``Insert``, ``Write``, and the pure-Python diff applier).  It lives
    here, in the module every tool already imports, so the four callers
    cannot drift apart -- and because ``base`` imports nothing from
    ``tools``, putting it here is also the only placement that avoids a
    circular import: ``filesystem.py`` re-imports ``edit``/``write``/
    ``insert`` at its bottom, so a helper defined *there* would make
    ``edit`` import a half-initialised ``filesystem``.

    A plain ``open(path, "w")`` TRUNCATES the file before writing, so a
    write that fails partway through -- ENOSPC, a quota, an I/O error,
    the process being killed -- left the file empty and the user's
    content unrecoverable while the tool returned a tidy "Error: ..."
    string.  Instead the new content goes to a sibling temp file which is
    then renamed over the target, mirroring the tmp-write + ``os.replace``
    discipline in ``SessionPersistence.save``.  ``os.replace`` is atomic:
    a reader sees either the complete old file or the complete new one,
    and ANY failure before it leaves PATH untouched.  The temp file is
    removed on the error path.

    Deliberate details:

    - ``errors="surrogateescape"`` and ``newline=""`` match the tools'
      read side, so invalid UTF-8 bytes (Latin-1, GBK, ...) round-trip
      and the content is written byte-for-byte instead of having "\\n"
      translated to ``os.linesep`` on Windows.
    - The temp name carries a random suffix rather than a fixed ".tmp":
      concurrent sub-agents can edit the same path, and a shared name
      would let one writer rename the other's half-written file into
      place.
    - An existing file's permission bits are copied onto the
      replacement, so editing an executable script does not drop its
      ``+x``.  A file that does not exist yet keeps the umask-derived
      permissions a plain ``open(path, "w")`` would have produced.  Only
      the "no such file" case is tolerated: any other stat/chmod failure
      propagates rather than silently shipping the wrong mode.

    There is deliberately no ``fsync``: the failures this guards against
    are process-level (a failed write, a kill, Ctrl-C), and in all of
    them the rename never happens.  Surviving a power loss in the window
    between write and rename would need an fsync here and on the parent
    directory, which ``SessionPersistence.save`` does not do either.

    Consequences of replacing the file rather than rewriting it in place,
    all inherent to the atomic-replace approach:

    - PATH must already be symlink-resolved.  ``os.replace`` onto a
      symlink overwrites the LINK with a regular file, where an in-place
      write would have followed it.  Every caller resolves the real path
      first (``Edit``/``Insert``/``Write`` via ``os.path.realpath`` on
      entry, ``_apply_section`` on its resolved target).
    - Hard links are broken: the new content lands on a new inode, so
      other links to the old inode keep the old content.
    - Owner/group, ACLs, SELinux labels and chattr flags are NOT carried
      over (only the mode bits are).  This matters mainly for an agent
      running as root over files owned by someone else.
    - On a crash between the write and the rename the temp file survives
      as ``<name>.<hex>.tmp``; the original is still intact, but the
      stray file is visible to Glob/Grep and to ``git status``.

    Raises OSError on failure.  Note this needs write permission on the
    DIRECTORY, not just on the file -- the one behavioural difference
    from a truncating in-place write (GNU ``patch``, used by ``Edit``'s
    diff mode, has always had the same requirement).
    """
    try:
        mode: int | None = stat.S_IMODE(os.stat(path).st_mode)
    except FileNotFoundError:
        mode = None  # brand-new file: keep the umask default from write_text
    tmp = f"{path}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        Path(tmp).write_text(content, encoding="utf-8", errors="surrogateescape", newline="")
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    except OSError:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


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
