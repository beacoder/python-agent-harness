"""Mkdir tool: create a directory (including parents)."""

from __future__ import annotations

import os

from .base import Tool, ToolContext, resolve_path


class Mkdir(Tool):
    name = "Mkdir"
    description = "Create a new directory (including parents)."
    parameters = {
        "type": "object",
        "properties": {
            "parent": {"type": "string", "description": "Parent directory"},
            "name": {"type": "string", "description": "Directory name to create"},
        },
        "required": ["parent", "name"],
    }

    def run(self, args: dict, ctx: ToolContext) -> str:
        parent = args["parent"]
        name = args["name"]
        # Only PARENT is a user path (``~``-expanded, project-relative);
        # NAME is a literal directory name, so a name like "~" stays "./~".
        path = os.path.realpath(os.path.join(resolve_path(parent, ctx.cwd), name))
        try:
            os.makedirs(path, exist_ok=True)
            return f"Directory {name} created/verified in {parent}"
        except OSError as e:
            return f"Error: {e}"
