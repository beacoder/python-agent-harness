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
        content = args.get("content")
        if content is None:
            return "Error: Required argument `content' missing"
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
                # surrogateescape: a non-UTF-8 file (Latin-1, GBK, ...) must
                # not make the read fail (UnicodeDecodeError used to escape
                # this handler and abort the whole write); the old content
                # is only used for the recorded diff
                with open(path, encoding="utf-8", errors="surrogateescape", newline="") as f:
                    old_content = f.read()
            except OSError:
                old_content = ""
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            # newline="": write the model's content byte-for-byte.  The
            # default text-mode write translates "\n" to os.linesep on
            # Windows, which (a) made the written bytes platform-dependent
            # and (b) broke the symmetric byte-exact read above (the same
            # logical content no longer compared equal).  surrogateescape
            # mirrors the read side; both together keep Write's behavior
            # identical on every platform.
            with open(path, "w", encoding="utf-8", errors="surrogateescape", newline="") as f:
                f.write(content)
        except OSError as e:
            return f"Error: {e}"
        diff_text = unified_diff(old_content, content, path)
        if diff_text:
            ctx.record_diff(diff_text)
        return f"Created file {filename} in {dir_path}"
