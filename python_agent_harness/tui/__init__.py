"""TUI package for the agent harness.

Re-exports the public API so that ``from python_agent_harness.tui import Tui``
and other imports continue to work unchanged after the split into
submodules.
"""

from __future__ import annotations

from .commands import CommandMixin
from .core import Tui
from .input import (
    SLASH_COMMANDS,
    SlashCompleter,
    UiQuestion,
    _custom_slash_commands,
    _history_path,
    _make_key_bindings,
    _make_prompt_session,
    _resolve_keyed_choice,
    _resolve_numbered_choice,
)
from .render import (
    _FC_DANGLING_RE,
    _FC_HEADER,
    _FC_LABEL,
    _FINAL_CHECK_RE,
    ASSISTANT_STYLE,
    SPINNER_FRAMES,
    USER_STYLE,
    _head_chars,
    _head_lines,
    _is_injected_user_text,
    _strip_final_check,
    _strip_reasoning,
    _tail_chars,
    _tail_lines,
    _tool_result_preview,
)

__all__ = [
    "CommandMixin",
    "Tui",
    "UiQuestion",
    "SlashCompleter",
    "SLASH_COMMANDS",
    "SPINNER_FRAMES",
    "USER_STYLE",
    "ASSISTANT_STYLE",
    "_resolve_keyed_choice",
    "_resolve_numbered_choice",
    "_make_key_bindings",
    "_make_prompt_session",
    "_history_path",
    "_custom_slash_commands",
    "_strip_final_check",
    "_strip_reasoning",
    "_is_injected_user_text",
    "_tail_lines",
    "_tail_chars",
    "_head_lines",
    "_head_chars",
    "_tool_result_preview",
    "_FINAL_CHECK_RE",
    "_FC_DANGLING_RE",
    "_FC_HEADER",
    "_FC_LABEL",
]
