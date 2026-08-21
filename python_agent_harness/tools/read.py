"""Read tool: whole-file reads with a size limit, streamed line ranges.

Read mirrors `gptel-agent--read-file-lines`: whole-file reads are
refused above READ_SIZE_LIMIT (400 KB, matching
`gptel-agent-read-file-size-threshold`), and line ranges are streamed
instead of loading the whole file into memory.  Oversized range
results are spilled to a temp file like Glob/Grep results, so nothing
is ever silently truncated.
"""

from __future__ import annotations

import os

from .base import Tool, ToolContext
from .filesystem import READ_SIZE_LIMIT, _spool


class Read(Tool):
    name = "Read"
    description = (
        "Read file contents between specified line numbers `start_line` and "
        "`end_line`, with both ends included.\n\n"
        'Consider using the "Grep" tool to find the right range to read first.\n\n'
        "Reads the whole file if the line range is not provided.\n\n"
        f"Files over {READ_SIZE_LIMIT // 1024} KB in size can only be read by "
        "specifying a line range.\n"
        "Very large line ranges are spilled to a temp file (see the 'Stored "
        "in:' path); use Read to view the full output."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "The path to the file to be read"},
            "start_line": {
                "type": "integer",
                "description": "The line to start reading from, defaults to the start of the file",
            },
            "end_line": {
                "type": "integer",
                "description": "The line up to which to read, defaults to the end of the file.",
            },
        },
        "required": ["file_path"],
    }

    def run(self, args: dict, ctx: ToolContext) -> str:
        path = args["file_path"]
        full = os.path.realpath(os.path.abspath(path))
        if os.path.isdir(full):
            return f"Error: cannot read {path}: is a directory"
        try:
            size = os.path.getsize(full)
        except OSError as e:
            return f"Error: cannot read {path}: {e}"
        start = args.get("start_line")
        end = args.get("end_line")
        if start is None and end is None:
            if size > READ_SIZE_LIMIT:
                return (
                    f"Error: File is too large ({size // 1024} KB > "
                    f"{READ_SIZE_LIMIT // 1024} KB). Please specify a line "
                    "range to read"
                )
            try:
                with open(full, encoding="utf-8", errors="replace") as f:
                    return f.read()
            except OSError as e:
                return f"Error: cannot read {path}: {e}"
        start = int(start or 1)
        if start < 1:
            start = 1
        if end is not None:
            end = int(end)
            if start > end:
                return f"Error: start_line {start} > end_line {end}"
        # Stream the file line by line instead of loading it whole, so
        # huge files can be read in ranges with constant memory.
        selected: list[str] = []
        total: int | None = None  # exact line count once EOF is reached
        reached_eof = True
        try:
            with open(full, encoding="utf-8", errors="replace") as f:
                for lineno, line in enumerate(f, 1):
                    if end is not None and lineno > end:
                        reached_eof = False
                        break
                    total = lineno
                    if lineno >= start:
                        selected.append(line)
        except OSError as e:
            return f"Error: cannot read {path}: {e}"
        if end is not None and reached_eof and total is not None:
            end = min(end, total)  # clamp to the real file end
        if total is None:
            total = 0  # empty file
        if start > total:
            return f"Error: start_line {start} > end_line {end if end is not None else total}"
        end_eff = end if end is not None else total
        if total is not None and reached_eof:
            header = f"Showing lines {start}-{end_eff} of {total}:\n\n"
        else:
            header = f"Showing lines {start}-{end_eff}:\n\n"
        return _spool(header + "".join(selected), "read")
