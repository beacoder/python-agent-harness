"""Windows Glob tool: ``git ls-files`` inside git repos, ``pathlib`` outside.

Windows ships neither ``tree`` nor ``find``, so the non-git fallback
in :class:`GlobTool` (which shells out to ``tree``) and the macOS
variant :class:`GlobMac` (which shells out to ``find``) both fail on
a stock Windows install.  ``GlobWindows`` replaces that fallback with
a pure-Python :func:`os.walk` approach: Python handles directory
traversal, pattern matching, and mtime sorting.

This is slower than the C-based ``tree``/``find`` on large directory
trees, but produces identical results and has no external dependencies
— critical on Windows where neither binary is guaranteed to exist.

Only the non-git fallback differs; the git path, the tool name, and
the result format are inherited unchanged so callers, the tool
registry, and the plan-mode write guard are platform-independent.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from .base import ToolContext
from .filesystem import _glob_to_regex, _natnump, _spool, _walk_files
from .glob import GlobTool


class GlobWindows(GlobTool):
    """Glob with a fault-tolerant pure-Python non-git fallback for Windows."""

    def _walk_fallback(self, pattern: str, base: str, depth: object) -> str:
        """Walk the tree with :func:`os.walk`, match, and sort by mtime.

        Matches files against a pathlib-style glob translated via
        :func:`_glob_to_regex`, filters hidden directories (``.git``,
        etc.), and sorts results by modification time (newest first),
        matching the ``tree --sort=mtime`` order of the Linux fallback.
        Unlike ``Path.rglob`` (whose OSError suppression only exists on
        3.13+), this traversal tolerates races and unreadable directories
        identically on every supported Python version.
        """
        root = Path(base)
        max_depth = depth if _natnump(depth) else None
        # pathlib rglob semantics: a pattern without a directory part is
        # matched against the basename at any depth; a pattern with one is
        # matched against the path relative to the root.
        if "/" in pattern or os.sep in pattern:
            rx = re.compile(_glob_to_regex(pattern.replace(os.sep, "/")))
        else:
            rx = re.compile(r"(?:.*/)?" + _glob_to_regex(pattern))
        errors: list[str] = []

        def onerror(e: OSError) -> None:
            errors.append(str(e))

        matches: list[tuple[float, str]] = []
        for p in _walk_files(root, onerror=onerror):
            rel = p.relative_to(root)
            rel_parts = rel.parts
            if any(part.startswith(".") for part in rel_parts[:-1]):
                continue
            if max_depth is not None and len(rel_parts) > max_depth:
                continue
            if not rx.fullmatch(rel.as_posix()):
                continue
            try:
                mtime = p.stat().st_mtime
            except OSError:
                mtime = 0.0
            matches.append((mtime, str(p)))

        if not matches and errors:
            return f"Error: {errors[0]}"

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
            path = os.path.expanduser(path)
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

        return self._walk_fallback(pattern, base, depth)
