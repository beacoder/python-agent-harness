"""Write tool: create/overwrite a file, recording a diff for the UI."""

from __future__ import annotations

import os

from ..diffrender import unified_diff
from .base import Tool, ToolContext


class Write(Tool):
    name = "Write"
    description = (
        "Create a new file with the given content. Overwrites an existing file — use with care!"
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory for the file"},
            "filename": {"type": "string", "description": "File name"},
            "content": {"type": "string", "description": "Full file content"},
        },
        "required": ["path", "filename", "content"],
    }

    def run(self, args: dict, ctx: ToolContext) -> str:
        dir_path = args.get("path") or "."
        filename = args.get("filename") or ""
        content = args.get("content") or ""
        # LLM may put the full file path in "filename" or in "path"
        if filename:
            path = os.path.realpath(os.path.abspath(os.path.join(dir_path, filename)))
        else:
            path = os.path.realpath(os.path.abspath(dir_path))
        if not filename:
            filename = os.path.basename(path) or os.path.basename(dir_path)
        existed = os.path.exists(path)
        old_content = ""
        if existed:
            try:
                with open(path, encoding="utf-8") as f:
                    old_content = f.read()
            except OSError:
                old_content = ""
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        except OSError as e:
            return f"Error: {e}"
        diff_text = unified_diff(old_content, content, path)
        if diff_text:
            ctx.record_diff(diff_text)
        return f"Created file {filename} in {dir_path}"
