"""Insert tool: insert text at a line number in an existing file.

Records a unified diff for the TUI so the change is visible in the
conversation panel like any other file edit.
"""

from __future__ import annotations

import os

from ..diffrender import unified_diff
from .base import Tool, ToolContext


class Insert(Tool):
    name = "Insert"
    description = (
        "Insert text at a specific line number in an existing file. "
        "line_number 0 = beginning, -1 = end."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path"},
            "line_number": {
                "type": "integer",
                "description": "Line after which to insert (0=start, -1=end)",
            },
            "new_str": {"type": "string", "description": "Text to insert"},
        },
        "required": ["path", "line_number", "new_str"],
    }

    def run(self, args: dict, ctx: ToolContext) -> str:
        path = os.path.realpath(os.path.abspath(args["path"]))
        try:
            with open(path, encoding="utf-8") as f:
                old_content = f.read()
                lines = old_content.splitlines(keepends=True)
        except OSError as e:
            return f"Error: cannot read {path}: {e}"
        ln = int(args["line_number"])
        new_str = args["new_str"]
        if not new_str.endswith("\n"):
            new_str += "\n"
        if ln < -1:
            return f"Error: line_number {ln} is invalid (use 0 for beginning, -1 for end)"
        if ln == -1 or ln >= len(lines):
            lines.append(new_str)
        elif ln == 0:
            lines.insert(0, new_str)
        else:
            lines.insert(ln, new_str)
        new_content = "".join(lines)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)
        except OSError as e:
            return f"Error: {e}"
        diff_text = unified_diff(old_content, new_content, path)
        if diff_text:
            ctx.record_diff(diff_text)
        return f"Successfully inserted text at line {ln} in {path}"
