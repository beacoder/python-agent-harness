"""Pure-Python unified diff applier (macOS Edit backend).

macOS ships Apple's old BSD ``/usr/bin/patch`` (patch 2.0-12u11-Apple),
which rejects well-formed hunks that GNU patch accepts: asymmetric
context (leading but no trailing context, or vice versa), hunks whose
last body line is a ``---``-rendered content line, etc.  Rather than
rewriting every diff into a shape that specific patch binary likes, the
macOS Edit tool applies diffs here, in Python:

- the diff is parsed STRUCTURALLY, so hunk-header counts are advisory
  (they are recounted from the body, like ``_fix_patch_headers``) and
  ``---``/``+++`` content lines are only file headers when followed by
  their ``+++``/``---`` partner;
- hunks are matched against the file by content with fuzz (context
  lines may mismatch up to ``FUZZ`` times, removed lines must match
  exactly), preferring the position the header claims;
- hunks are applied bottom-up so earlier positions stay valid;
- files are written only when every hunk of the section matched.

The applier is platform-independent, so its behavior is covered by
tests on every OS.
"""

from __future__ import annotations

import os
import re

_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_FILE_OLD_RE = re.compile(r"^---[ \t]")
_FILE_NEW_RE = re.compile(r"^\+\+\+[ \t]")
_HUNK_BOUNDARY_RE = re.compile(r"^@@ ")

# Maximum number of context lines that may mismatch when anchoring a
# hunk (removed lines must always match exactly).
FUZZ = 3


class _Hunk:
    __slots__ = ("old_start", "body")

    def __init__(self, old_start: int, body: list[tuple[str, str, bool]]) -> None:
        # body entries: (kind, content, no_newline); kind in ' ', '-', '+'.
        self.old_start = old_start
        self.body = body

    def old_seq(self) -> list[tuple[str, str]]:
        """(kind, content-without-newline) for the lines that must exist
        in the old file (' ' context and '-' removed lines)."""
        return [
            (kind, content.rstrip("\n")) for kind, content, _ in self.body if kind in (" ", "-")
        ]


class _Section:
    __slots__ = ("new_path", "hunks")

    def __init__(self, new_path: str, hunks: list[_Hunk]) -> None:
        self.new_path = new_path
        self.hunks = hunks


def _section_header_at(lines: list[str], idx: int) -> bool:
    """Whether lines[idx] begins a file section: a ``--- path`` line
    immediately followed by a ``+++ path`` line.  Removed/added content
    lines rendered ``---``/``+++`` inside a hunk body are only mistaken
    for headers when the pair shape matches, so the trailing ``+++``
    partner is required (mirrors ``_starts_file_section``)."""
    if idx + 1 >= len(lines):
        return False
    return bool(_FILE_OLD_RE.match(lines[idx]) and _FILE_NEW_RE.match(lines[idx + 1]))


def _parse(diff_text: str) -> list[_Section]:
    """Parse a unified diff into file sections (paths + hunks)."""
    lines = diff_text.splitlines(keepends=True)
    sections: list[_Section] = []
    i = 0
    n = len(lines)
    while i < n:
        if not _section_header_at(lines, i):
            i += 1
            continue
        new_path = lines[i + 1][len("+++") :].strip()
        i += 2
        hunks: list[_Hunk] = []
        while i < n:
            m = _HUNK_HEADER_RE.match(lines[i])
            if m:
                old_start = int(m.group(1))
                body: list[tuple[str, str, bool]] = []
                i += 1
                while (
                    i < n
                    and not _HUNK_BOUNDARY_RE.match(lines[i])
                    and not _section_header_at(lines, i)
                ):
                    line = lines[i]
                    if line.startswith("\\"):  # "\ No newline at end of file"
                        if body and body[-1][0] in (" ", "-", "+"):
                            kind, content, _ = body[-1]
                            body[-1] = (kind, content, True)
                    elif line[:1] in (" ", "-", "+"):
                        body.append((line[0], line[1:], False))
                    i += 1
                hunks.append(_Hunk(old_start, body))
                continue
            if _section_header_at(lines, i):
                break
            i += 1  # stray line (e.g. "diff --git", "index ...") — skip
        sections.append(_Section(new_path, hunks))
    return sections


def _candidate_positions(start: int, limit: int) -> list[int]:
    """Match offsets around *start* (0-based), closest first, within
    ``FUZZ`` lines and bounded by [0, *limit*]."""
    out: list[int] = []
    seen: set[int] = set()
    for d in range(FUZZ + 1):
        for p in (start - d, start + d):
            if 0 <= p <= limit and p not in seen:
                seen.add(p)
                out.append(p)
    return out


def _match_hunk(hunk: _Hunk, file_lines: list[str]) -> int | None:
    """Find the 0-based position where the hunk's old lines match.

    Removed lines must match exactly; up to ``FUZZ`` context lines may
    mismatch (patch-style fuzz).  Pure-insertion hunks (no old lines)
    anchor at ``old_start - 1``.
    """
    old_seq = hunk.old_seq()
    if not old_seq:
        p = hunk.old_start - 1
        return max(0, min(p, len(file_lines)))
    start = hunk.old_start - 1
    limit = len(file_lines) - len(old_seq)
    for p in _candidate_positions(start, limit):
        mismatches = 0
        ok = True
        for i, (kind, content) in enumerate(old_seq):
            file_content = file_lines[p + i].rstrip("\n")
            if kind == "-":
                if file_content != content:
                    ok = False
                    break
            elif file_content != content:
                mismatches += 1
                if mismatches > FUZZ:
                    ok = False
                    break
        if ok:
            return p
    return None


def _apply_hunk(hunk: _Hunk, pos: int, file_lines: list[str]) -> None:
    """Replace the matched region with the hunk's new lines in place."""
    old_count = sum(1 for kind, _, _ in hunk.body if kind in (" ", "-"))
    new_block: list[str] = []
    p = pos
    for kind, content, no_newline in hunk.body:
        if kind == " ":
            new_block.append(file_lines[p])
            p += 1
        elif kind == "-":
            p += 1
        else:  # "+"
            if no_newline:
                new_block.append(content.rstrip("\n"))
            elif content.endswith("\n"):
                new_block.append(content)
            else:
                new_block.append(content + "\n")
    file_lines[pos : pos + old_count] = new_block


def _resolve_target(new_path: str, cwd: str, fallback_path: str | None) -> str:
    """Resolve a diff's ``+++ path`` against *cwd*, with the single-file
    *fallback_path* as backup when the name does not match an existing
    file (e.g. the model wrote a bare basename for a deep path)."""
    name = new_path
    for prefix in ("a/", "b/"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    if os.path.isabs(name):
        return name
    target = os.path.join(cwd, name)
    if not os.path.isfile(target) and fallback_path and os.path.isfile(fallback_path):
        return fallback_path
    return target


def _apply_section(section: _Section, cwd: str, fallback_path: str | None) -> tuple[bool, str]:
    target = _resolve_target(section.new_path, cwd, fallback_path)
    if not os.path.isfile(target):
        return False, f"target file does not exist: {target}"
    try:
        with open(target, encoding="utf-8", errors="replace") as f:
            file_lines = f.readlines()
    except OSError as e:
        return False, f"cannot read {target}: {e}"
    # Match every hunk against the ORIGINAL content first; only write
    # when all matched (a failed hunk leaves the file untouched).
    plan: list[tuple[_Hunk, int]] = []
    for hunk in section.hunks:
        pos = _match_hunk(hunk, file_lines)
        if pos is None:
            return (
                False,
                f"hunk at line {hunk.old_start} failed to match the file content",
            )
        plan.append((hunk, pos))
    new_lines = list(file_lines)
    for hunk, pos in reversed(plan):  # bottom-up: earlier positions stay valid
        _apply_hunk(hunk, pos, new_lines)
    try:
        with open(target, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
    except OSError as e:
        return False, f"cannot write {target}: {e}"
    return True, f"patched {target}"


def apply_unified_diff(
    diff_text: str, cwd: str, fallback_path: str | None = None
) -> tuple[bool, str]:
    """Apply a unified diff to files under *cwd*.

    Returns ``(ok, message)``; on failure no file is written.  Multi-file
    diffs are applied section by section.  *fallback_path* is used when a
    section's target cannot be resolved to an existing file (single-file
    mode).
    """
    sections = _parse(diff_text)
    if not sections:
        return False, "no file sections found in diff"
    for section in sections:
        ok, msg = _apply_section(section, cwd, fallback_path)
        if not ok:
            return False, msg
    return True, f"applied {len(sections)} file section(s)"
