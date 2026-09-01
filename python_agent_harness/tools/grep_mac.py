"""macOS Grep tool: ``git grep -E`` instead of ``git grep -P``.

Apple's default Git (installed via Xcode Command Line Tools) is built
without PCRE support, so ``git grep -P`` (Perl-compatible regex) fails
with::

    fatal: cannot use Perl-compatible regexes when not compiled with USE_LIBPCRE

The Linux :class:`Grep` tool uses ``-P`` as the first-choice backend;
when that fails on macOS the fallback chain (``rg`` → ``grep``) still
works, but:

- ``rg`` (ripgrep) is not installed by default on macOS.
- BSD ``grep`` supports only POSIX ERE, losing Perl features
  (``\\d``, ``\\w``, lookaheads).
- Two unnecessary failed attempts add latency before reaching a
  working backend.

``GrepMac`` replaces ``-P`` with ``-E`` (POSIX extended regex) in the
``git grep`` command, so the git path — the fastest and most common
backend — works on stock macOS.  PCRE character class shorthands
(``\\d``, ``\\w``, ``\\s`` and their negations) are translated to
POSIX bracket expressions so the LLM's regexes produce the same
matches they would under ``-P``.  The translation is context-aware:
it skips shorthands inside existing character classes (``[\\d.]``)
and shorthands preceded by a literal backslash (``\\\\d``), so the
regex structure is never corrupted.  ``\\b`` (word boundary) is
translated to ``[[:<:]]`` / ``[[:>:]]`` on macOS git, which supports
these GNU-compatible extensions.

Features with no POSIX equivalent (lookaheads, backreferences) are
left as-is — they will cause a syntax error, which is better than
silently matching wrong content.

The fallback chain skips the parent's ``git grep -P`` attempt (guaranteed
to fail on macOS) and goes directly to ``rg`` / ``grep`` via the shared
:meth:`Grep._fallback_rg_grep` method.

Only the ``git grep`` regex flag differs; the tool name and result
format are inherited so callers, the tool registry, and the plan-mode
write guard are platform-independent.
"""

from __future__ import annotations

import os
import re
import subprocess

from .base import ToolContext
from .filesystem import _git_root
from .grep import Grep, _grep_out

_PCRE_CLASS_MAP: dict[str, str] = {
    r"\d": "[0-9]",
    r"\D": "[^0-9]",
    r"\w": "[A-Za-z0-9_]",
    r"\W": "[^A-Za-z0-9_]",
    r"\s": "[ \t\n\r\f\v]",
    r"\S": "[^ \t\n\r\f\v]",
}

# Match \d, \D, \w, \W, \s, \S, or \b that is NOT preceded by an
# odd number of backslashes (i.e. not escaped).  The lookbehind
# (?<!\\) ensures we don't match a shorthand that is itself escaped.
# The ``prefix`` group captures any preceding pairs of literal
# backslashes so they can be preserved in the replacement — without
# this, re.sub would discard them along with the full match.
_PCRE_SHORTHAND_RE = re.compile(r"(?<!\\)(?P<prefix>(?:\\\\)*)(?P<shorthand>\\[dDwWsSb])")


def _is_escaped(s: str, pos: int) -> bool:
    """True if the character at *pos* is preceded by an odd number of backslashes."""
    count = 0
    j = pos - 1
    while j >= 0 and s[j] == "\\":
        count += 1
        j -= 1
    return count % 2 == 1


def _find_char_class_end(s: str, start: int) -> int | None:
    """Find the closing ``]`` of a character class starting at *start*.

    Handles ``[]]`` (``]`` as first char) and ``[^]]``.  Returns the
    index of the closing ``]`` or ``None`` if unterminated.
    """
    i = start + 1
    if i < len(s) and s[i] == "^":
        i += 1
    if i < len(s) and s[i] == "]":
        i += 1
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            i += 2
            continue
        if s[i] == "]":
            return i
        i += 1
    return None


def _pcre_to_ere(regex: str) -> str:
    """Translate PCRE character class shorthands to POSIX ERE.

    Translates ``\\d``, ``\\w``, ``\\s``, their negations, and ``\\b``
    (word boundary) to POSIX-compatible equivalents.  The translation is
    context-aware:

    - Shorthands **inside** an existing character class (``[\\d.]``) are
      left untouched — replacing them would create nested ``[]`` which
      is invalid in POSIX ERE.  ``git grep -E`` on macOS handles
      ``\\d`` inside ``[]`` natively, so this is safe.
    - Shorthands preceded by a literal backslash (``\\\\d``) are left
      untouched — they represent a literal backslash + letter, not a
      character class.

    Features with no POSIX equivalent (lookaheads, backreferences) are
    left as-is so they produce a clear syntax error instead of silently
    matching wrong content.
    """
    # Find all character class spans so we can skip shorthands inside them.
    char_class_spans: list[tuple[int, int]] = []
    i = 0
    while i < len(regex):
        if regex[i] == "[" and not _is_escaped(regex, i):
            end = _find_char_class_end(regex, i)
            if end is not None:
                char_class_spans.append((i, end))
                i = end + 1
                continue
        i += 1

    def _in_char_class(pos: int) -> bool:
        return any(start < pos < end for start, end in char_class_spans)

    def _replace(m: re.Match) -> str:
        if _in_char_class(m.start()):
            return m.group(0)
        prefix = m.group("prefix")
        shorthand = m.group("shorthand")
        if shorthand == r"\b":
            pos = m.start("shorthand")
            after = regex[pos + 2] if pos + 2 < len(regex) else ""
            before = regex[pos - 1] if pos > 0 else ""
            if before and (before.isalnum() or before == "_"):
                return prefix + "[[:>:]]"
            if after and (after.isalnum() or after == "_"):
                return prefix + "[[:<:]]"
            return prefix + "[[:<:]]"
        return prefix + _PCRE_CLASS_MAP.get(shorthand, shorthand)

    return _PCRE_SHORTHAND_RE.sub(_replace, regex)


class GrepMac(Grep):
    """Grep with ``git grep -E`` for macOS (no PCRE dependency)."""

    def run(self, args: dict, ctx: ToolContext) -> str:
        regex = args["regex"]
        path = os.path.realpath(args["path"])
        if not os.path.isdir(path) and not os.path.isfile(path):
            return f"Error: path {args['path']} is not readable"
        glob = args.get("glob")
        context = args.get("context_lines")
        if context is not None:
            context = max(0, min(15, int(context)))

        git_root = _git_root(path)
        if git_root:
            rel = os.path.relpath(path, git_root)
            pathspec = rel
            if glob and os.path.isdir(path):
                pathspec = os.path.join(rel, glob).replace(os.sep, "/")
            ere_regex = _pcre_to_ere(regex)
            cmd = [
                "git",
                "grep",
                "--line-number",
                "--no-color",
                "--max-count=1000",
                "--untracked",
                "-E",
                "-e",
                ere_regex,
                "--",
                pathspec,
            ]
            if context:
                cmd = cmd[:3] + [f"-C{context}"] + cmd[3:]
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=git_root,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=60,
                )
            except (OSError, subprocess.TimeoutExpired):
                proc = None
            if proc is not None and proc.returncode in (0, 1):
                return _grep_out(proc, "git")

        # Non-git or git grep -E failed: skip the parent's git grep -P
        # attempt (guaranteed to fail on macOS) and go directly to the
        # shared rg/grep fallback chain.
        return self._fallback_rg_grep(regex, path, glob, context)
