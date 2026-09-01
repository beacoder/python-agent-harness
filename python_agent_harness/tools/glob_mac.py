"""macOS Glob tool: ``git ls-files`` inside git repos, ``find`` outside.

Apple ships neither GNU ``tree`` nor a compatible equivalent, so the
non-git fallback in :class:`GlobTool` (which shells out to ``tree``)
fails on a stock macOS install.  ``GlobMac`` replaces that fallback
with a hybrid approach: ``find`` (C binary, pre-installed on macOS)
handles directory traversal and pattern matching, while Python handles
mtime sorting — which ``find`` does not support.

This is significantly faster than the previous pure-Python
:meth:`pathlib.Path.rglob` approach on large directory trees, while
producing identical results.

Only the non-git fallback differs; the git path, the tool name, and
the result format are inherited unchanged so callers, the tool
registry, and the plan-mode write guard are platform-independent.
"""

from __future__ import annotations

import os
import subprocess

from .base import ToolContext
from .filesystem import _natnump, _spool
from .glob import GlobTool


class GlobMac(GlobTool):
    """Glob with a hybrid ``find`` + Python-sort non-git fallback for macOS."""

    def _find_fallback(self, pattern: str, base: str, depth: object) -> str:
        """Use ``find`` for traversal + matching, Python for mtime sort.

        ``find`` (C binary) walks the directory tree and applies pattern
        matching natively, which is significantly faster than
        :meth:`pathlib.Path.rglob` on large trees.  Results are then
        sorted by modification time (newest first) in Python, matching
        the ``tree --sort=mtime`` order of the Linux fallback.  Hidden
        directories (``.git``, etc.) are skipped via ``-path */.* -prune``.
        Symlinks are followed (``-L``), matching ``tree -l`` and the old
        :meth:`pathlib.Path.rglob` behavior.
        """
        cmd = [
            "find",
            "-L",
            base,
            "-path",
            "*/.*",
            "-prune",
            "-o",
            "-type",
            "f",
            "-iname",
            pattern,
            "-print",
        ]
        if _natnump(depth):
            cmd[3:3] = ["-maxdepth", str(depth)]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            return f"Error: {e}"
        lines = [line for line in proc.stdout.splitlines() if line]
        if not lines:
            if proc.returncode != 0:
                out = f"Glob failed with exit code {proc.returncode}\n"
                out += proc.stderr or ""
                return _spool(out, "glob")
            return ""

        matches: list[tuple[float, str]] = []
        for path in lines:
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                mtime = 0.0
            matches.append((mtime, path))

        matches.sort(key=lambda t: t[0], reverse=True)
        out = "\n".join(path for _, path in matches)
        if not out:
            return ""
        return _spool(out + "\n", "glob")

    def run(self, args: dict, ctx: ToolContext) -> str:
        pattern = args.get("pattern") or ""
        if not pattern:
            return "Error: pattern must not be empty"
        path = args.get("path")
        if path:
            if not (os.path.isdir(path) and os.access(path, os.R_OK)):
                return f"Error: path {path} is not readable"
        else:
            path = ctx.cwd
        base = os.path.realpath(path)
        depth = args.get("depth")

        from .filesystem import _git_root

        git_root = _git_root(base)

        if git_root:
            # The git path is identical to the parent — delegate.
            return super().run(args, ctx)

        # Non-git: hybrid find + Python sort fallback instead of `tree`.
        return self._find_fallback(pattern, base, depth)
