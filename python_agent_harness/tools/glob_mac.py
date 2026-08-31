"""macOS Glob tool: ``git ls-files`` inside git repos, ``pathlib`` outside.

Apple ships neither GNU ``tree`` nor a compatible equivalent, so the
non-git fallback in :class:`GlobTool` (which shells out to ``tree``)
fails on a stock macOS install.  ``GlobMac`` replaces that fallback
with :meth:`pathlib.Path.rglob`, which is pure-Python, requires no
external binary, and produces identical results.

Only the non-git fallback differs; the git path, the tool name, and
the result format are inherited unchanged so callers, the tool
registry, and the plan-mode write guard are platform-independent.
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path

from .base import ToolContext
from .filesystem import _natnump, _spool
from .glob import GlobTool


class GlobMac(GlobTool):
    """Glob with a pure-Python non-git fallback for macOS."""

    def _tree_fallback(self, pattern: str, base: str, depth: object) -> str:
        """Walk *base* with :func:`pathlib.Path.rglob` instead of ``tree``.

        Results are absolute paths sorted by modification time (newest
        first), matching the ``tree --sort=mtime`` order of the Linux
        fallback.  Hidden directories (``.git``, etc.) are skipped.
        """
        root = Path(base)
        matches: list[tuple[float, str]] = []
        if _natnump(depth):
            # Depth-limited: walk manually so we can count levels.
            for dirpath, dirnames, filenames in os.walk(base):
                # Skip hidden directories (mirrors tree's -I .git and
                # the general expectation that dotfiles are excluded).
                dirnames[:] = [d for d in dirnames if not d.startswith(".")]
                rel = os.path.relpath(dirpath, base)
                level = 0 if rel == "." else rel.count(os.sep) + 1
                if level >= depth:
                    dirnames.clear()
                    continue
                for name in filenames:
                    if fnmatch.fnmatch(name.lower(), pattern.lower()):
                        full = os.path.join(dirpath, name)
                        try:
                            mtime = os.path.getmtime(full)
                        except OSError:
                            mtime = 0.0
                        matches.append((mtime, full))
        else:
            # Unlimited depth: rglob is simpler.
            for p in root.rglob("*"):
                if any(part.startswith(".") for part in p.relative_to(root).parts):
                    continue
                if p.is_file() and fnmatch.fnmatch(p.name.lower(), pattern.lower()):
                    try:
                        mtime = p.stat().st_mtime
                    except OSError:
                        mtime = 0.0
                    matches.append((mtime, str(p)))

        # Sort newest-first (like tree --sort=mtime).
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

        # Non-git: pure-Python fallback instead of `tree`.
        return self._tree_fallback(pattern, base, depth)
