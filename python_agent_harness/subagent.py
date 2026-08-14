"""Sub-agent runner: delegated agent tasks with error containment.

Mirrors gptel-agent-harness-agent.el: unexpected response shapes become
error strings fed back to the parent instead of crashing it.  In plan
mode, sub-agents receive the read-only reminder.
"""

from __future__ import annotations

from . import config
from .agent import run_agent_loop
from .models import Message
from .prompts import load_agent_prompt


def _subagent_system_prompt(session: object) -> str | None:
    """The sub-agent's system prompt: its OWN prompt only.

    Never falls back to the parent's `system_prompt` (which carries the
    parent's project context and task-completion rules) — a sub-agent
    must not inherit any context from the parent.  When the session has
    no sub-agent prompt configured, the default bundled one is used.
    """
    own = getattr(session, "subagent_system_prompt", None)
    if own:
        return own
    return load_agent_prompt(config.DEFAULT_SUBAGENT_PROMPT_FILE)


def run_subagent(
    parent_session: object,
    description: str,
    prompt: str,
    client: object | None = None,
) -> str:
    """Run a sub-agent task; return a result string (never raises).

    ``client`` (when given) is the per-invocation dedicated client
    (see ``AgentSession.run_subagent``); the loop falls back to the
    session's shared sub-agent client otherwise.
    """
    session = parent_session
    try:
        messages = [Message(role="user", content=prompt)]
        # NOTE: the plan-mode read-only reminder is injected by the agent
        # loop itself (`AgentLoop._inject_pending_prompts`, once per
        # sub-agent loop) — do NOT insert it here as well, or it appears
        # twice in the request.
        result = run_agent_loop(
            session=session,
            messages=messages,
            top_level=False,
            system=_subagent_system_prompt(session),
            max_rounds=config.SUBAGENT_MAX_ROUNDS,
            client=client,
        )
        if isinstance(result, str):
            return result
        return f"Error: Task {description!r} returned an unexpected response {result!r}"
    except Exception as e:  # noqa: BLE001 - containment boundary
        return f'Error: Task "{description}" returned an unexpected response — {e}'
