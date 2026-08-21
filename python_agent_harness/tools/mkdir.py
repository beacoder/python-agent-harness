"""Mkdir tool: create a directory (including parents)."""

from __future__ import annotations

import os

from .base import Tool, ToolContext


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
        path = os.path.realpath(os.path.abspath(os.path.join(parent, name)))
        try:
            os.makedirs(path, exist_ok=True)
            return f"Directory {name} created/verified in {parent}"
        except OSError as e:
            return f"Error: {e}"
