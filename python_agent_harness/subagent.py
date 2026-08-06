"""Sub-agent runner: delegated agent tasks with error containment.

Mirrors gptel-agent-harness-agent.el: unexpected response shapes become
error strings fed back to the parent instead of crashing it.  In plan
mode, sub-agents receive the read-only reminder.
"""

from __future__ import annotations

from . import config
from .agent import run_agent_loop
from .models import Message


def run_subagent(
    parent_session: object,
    description: str,
    prompt: str,
) -> str:
    """Run a sub-agent task; return a result string (never raises)."""
    session = parent_session
    try:
        messages = [Message(role="user", content=prompt)]
        # NOTE: the plan-mode read-only reminder is injected by the agent
        # loop itself (`AgentLoop._inject_pending_prompts`, once per
        # sub-agent FSM) — do NOT insert it here as well, or it appears
        # twice in the request.
        result = run_agent_loop(
            session=session,
            messages=messages,
            top_level=False,
            system=getattr(session, "subagent_system_prompt", None)
            or session.system_prompt,
            max_rounds=config.SUBAGENT_MAX_ROUNDS,
        )
        if isinstance(result, str):
            return result
        return (
            f"Error: Task {description!r} returned an unexpected response "
            f"{result!r}"
        )
    except Exception as e:  # noqa: BLE001 - containment boundary
        return (
            f'Error: Task "{description}" returned an unexpected response '
            f"— {e}"
        )
