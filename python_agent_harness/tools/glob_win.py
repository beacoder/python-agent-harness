"""Windows Glob tool: ``git ls-files`` inside git repos, ``pathlib`` outside.

Windows ships neither ``tree`` nor ``find``, so the non-git fallback
in :class:`GlobTool` (which shells out to ``tree``) and the macOS
variant :class:`GlobMac` (which shells out to ``find``) both fail on
a stock Windows install.  ``GlobWindows`` replaces that fallback with
a pure-Python :meth:`pathlib.Path.rglob` approach: Python handles
directory traversal, pattern matching, and mtime sorting.

This is slower than the C-based ``tree``/``find`` on large directory
trees, but produces identical results and has no external dependencies
— critical on Windows where neither binary is guaranteed to exist.

Only the non-git fallback differs; the git path, the tool name, and
the result format are inherited unchanged so callers, the tool
registry, and the plan-mode write guard are platform-independent.
"""

from __future__ import annotations

import os
from pathlib import Path

from .base import ToolContext
from .filesystem import _natnump, _spool
from .glob import GlobTool


class GlobWindows(GlobTool):
    """Glob with a pure-Python ``pathlib.rglob`` non-git fallback for Windows."""

    def _rglob_fallback(self, pattern: str, base: str, depth: object) -> str:
        """Use ``pathlib.Path.rglob`` for traversal + matching, Python for mtime sort.

        Walks the directory tree with :meth:`pathlib.Path.rglob`, filters
        hidden directories (``.git``, etc.), and sorts results by
        modification time (newest first), matching the ``tree --sort=mtime``
        order of the Linux fallback.  Symlinks are followed by default
        via ``Path.rglob``.
        """
        root = Path(base)
        max_depth = depth if _natnump(depth) else None

        matches: list[tuple[float, str]] = []
        try:
            for p in root.rglob(pattern):
                if not p.is_file():
                    continue
                # skip hidden directories (.git, etc.)
                if any(part.startswith(".") for part in p.relative_to(root).parts[:-1]):
                    continue
                if max_depth is not None:
                    rel_depth = len(p.relative_to(root).parts)
                    if rel_depth > max_depth:
                        continue
                try:
                    mtime = p.stat().st_mtime
                except OSError:
                    mtime = 0.0
                matches.append((mtime, str(p)))
        except OSError as e:
            return f"Error: {e}"

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
            return super().run(args, ctx)

        return self._rglob_fallback(pattern, base, depth)
