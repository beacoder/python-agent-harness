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
backend — works on stock macOS.  The rest of the fallback chain (rg,
grep) is inherited unchanged.

Only the ``git grep`` regex flag differs; the tool name and result
format are inherited so callers, the tool registry, and the plan-mode
write guard are platform-independent.
"""

from __future__ import annotations

import os
import subprocess

from .base import ToolContext
from .filesystem import _git_root
from .grep import Grep, _grep_out


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

        git_root = _git_root(path) if os.path.isdir(path) else None
        if git_root:
            rel = os.path.relpath(path, git_root)
            pathspec = rel
            if glob and os.path.isdir(path):
                pathspec = os.path.join(rel, glob).replace(os.sep, "/")
            # -E (extended regex) instead of -P (Perl regex): Apple's
            # Git is built without PCRE, so -P fails on stock macOS.
            cmd = [
                "git",
                "grep",
                "--line-number",
                "--no-color",
                "--max-count=1000",
                "--untracked",
                "-E",
                "-e",
                regex,
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

        # Non-git or git grep failed: delegate to the parent's rg/grep
        # fallback chain (identical on both platforms).
        return super().run(args, ctx)
