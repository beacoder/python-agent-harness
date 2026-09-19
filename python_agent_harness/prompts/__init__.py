"""Prompt assembly and agent/skill discovery.

Markdown prompt assets (``agent.md``, ``commands/``, ``agents/``) live
in this package alongside the code in ``core.py``.
"""

from .core import (  # noqa: F401
    _FRONTMATTER_RE,
    _SKILLS_FALLBACK,
    _SKILLS_PLACEHOLDER_RE,
    _TOOL_INSTRUCTIONS_PLACEHOLDER_RE,
    AGENTS_DIR,
    PROMPTS_DIR,
    RESERVED_AGENT_NAME,
    _agent_frontmatter,
    _agent_name_from_file,
    _find_up,
    _git_toplevel,
    _image_placeholder_text,
    _is_mode_reminder_text,
    _is_plan_exit_notice,
    _message_role,
    _message_text,
    _parse_skill_frontmatter,
    _tool_excluded,
    agent_exclude_tools,
    assemble_agent_prompt,
    assemble_tool_instructions,
    compact_summary,
    compacted_messages,
    discover_agents,
    discover_skills,
    find_agents_md_files,
    index_skills,
    load_agent_prompt,
    load_context_files,
    load_task_completion_rules,
    read_prompt_file,
    strip_frontmatter,
    user_prompt_texts,
)
