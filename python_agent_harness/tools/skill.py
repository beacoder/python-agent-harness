"""Skill tool: load a skill into the current conversation."""

from __future__ import annotations

import os
from pathlib import Path

from .base import Tool, ToolContext


class Skill(Tool):
    name = "Skill"
    description = (
        "Load a skill to get detailed instructions for a specific task. "
        "Use this when a task matches an available skill's description. "
        "Invoke with the skill name and optional args."
    )
    parameters = {
        "type": "object",
        "properties": {
            "skill": {"type": "string", "description": "Name of the skill to load"},
            "args": {"type": "string", "description": "Optional arguments for the skill"},
        },
        "required": ["skill"],
    }

    def run(self, args: dict, ctx: ToolContext) -> str:
        name = args.get("skill", "")
        skill_path = ctx.find_skill(name)
        if not skill_path:
            return f"Error: skill {name!r} not found"
        text = Path(skill_path).read_text(encoding="utf-8", errors="replace")
        return f"[Skill: {name}]\n\n{text}"
