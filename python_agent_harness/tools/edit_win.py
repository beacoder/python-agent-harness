"""Windows Edit tool: pure-Python diff applier (no ``patch`` binary).

Windows does not ship ``patch.exe``, so the diff/patch mode of
:class:`Edit` (which shells out to ``patch --forward``) fails on a
stock Windows install.  The pure-Python applier (``diffapply.py``)
parses the diff structurally and matches hunks with fuzz, so the same
diffs apply on Windows, macOS, and Linux.

Only the diff-application strategy differs; the tool name stays
``Edit`` so callers, the tool registry and the plan-mode write guard
are platform-independent.

The implementation is identical to :class:`EditMac` — both platforms
lack a reliable ``patch`` binary and use the same Python applier.
"""

from __future__ import annotations

import os

from ..diffrender import unified_diff
from .base import ToolContext
from .diffapply import apply_unified_diff
from .edit import Edit, _strip_diff_fence


class EditWindows(Edit):
    """Edit with a pure-Python diff/patch backend for Windows."""

    def _apply_patch(self, path: str, cwd: str, diff: str, ctx: ToolContext) -> str:
        text = diff if diff.endswith("\n") else diff + "\n"
        text = _strip_diff_fence(text)
        is_file = os.path.isfile(path)
        old_content = None
        if is_file:
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    old_content = f.read()
            except OSError:
                old_content = None
        ok, msg = apply_unified_diff(text, cwd, path if is_file else None)
        if not ok:
            return f"Error: Failed to apply diff to {path}.\n{msg}"
        if old_content is not None and os.path.isfile(path):
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    new_content = f.read()
                diff_text = unified_diff(old_content, new_content, path)
                if diff_text:
                    ctx.record_diff(diff_text)
            except OSError:
                pass
        return (
            f"Diff successfully applied to {path}.\n"
            f"Applied with the built-in Python diff applier.\n{msg}"
        )
