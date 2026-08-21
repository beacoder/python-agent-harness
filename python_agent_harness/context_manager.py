"""Context management for the agent loop: ratio tracking + compaction.

Extracted from agent.py (no logic changes): the FSM driver delegates
``_update_context_ratio`` / ``_need_compaction`` / ``compact`` to
``ContextManager``, which reads/writes the loop's shared state
(``messages`` / ``session``) through the loop reference.
"""

from __future__ import annotations

from typing import Any

from . import config
from .prompts import compact_summary, compacted_messages, user_prompt_texts


class ContextManager:
    """Context-ratio tracking and compaction for one agent loop.

    ``update_context_ratio`` receives the two token-estimator functions
    from the loop's delegate so the call site keeps resolving them
    through the ``agent`` module namespace (tests patch
    ``python_agent_harness.agent.estimate_payload_tokens``).
    """

    def __init__(self, loop: Any) -> None:
        self.loop = loop

    def update_context_ratio(
        self,
        estimate_payload_tokens: Any,
        context_window_for: Any,
    ) -> None:
        loop = self.loop
        raw = estimate_payload_tokens(
            loop.system,
            [m.to_api() for m in loop.messages],
            [t.to_api() for t in loop.session.tool_specs()],
        )
        loop.session.calibrator.last_raw_estimate = raw
        calibrated = loop.session.calibrator.calibrate(raw)
        window = context_window_for(loop.session.model)
        loop.session.context_ratio = calibrated / float(window)
        loop.session.notify("context")

    def need_compaction(self) -> bool:
        loop = self.loop
        return (
            loop.top_level
            and loop.session.tools_enabled
            and not loop.session.compacting
            and loop.session.context_ratio is not None
            and loop.session.context_ratio > config.CONTEXT_TRIGGER
        )

    def compact(self) -> bool:
        """Compact the conversation; return True on success.

        On success the history is replaced by the summary frame followed
        by every real user prompt (nudges and other harness-injected
        messages excluded), so the model keeps the actual requests; the
        last prompt is the resume request for the next round.
        """
        loop = self.loop
        prompts = user_prompt_texts(loop.messages)
        if not prompts:
            return False
        loop.session.compacting = True
        try:
            conversation = "\n\n".join(f"{m.role}: {m.text()}" for m in loop.messages if m.text())
            summary = compact_summary(
                loop.session.client, conversation, cancel_check=loop._is_cancelled
            )
            if not summary:
                return False
            # The summary replaces the whole conversation history EXCEPT
            # the system prompt (loop.system is passed separately and
            # stays untouched): it is part of the user turn, never a
            # system message.  Every real user prompt (nudges and other
            # harness-injected messages excluded) is preserved verbatim
            # after the frame, so the model keeps the actual requests.
            loop.messages = compacted_messages(summary, prompts)
            # The shared conversation now is the compacted one: mirror it
            # onto session.last_messages so the TUI (renders from it) and
            # a later manual /compact start from the summary, not the old
            # full history.
            if loop.top_level and not loop._is_cancelled():
                loop.session.last_messages = list(loop.messages)
            # Fresh start for the resumed conversation: the pre-compaction
            # nudge budget must not carry over, or the first terminal
            # answer after compaction ends the run immediately.
            loop.supervisor.reset_nudges()
            loop.session.notify("compact")
            return True
        except Exception as e:  # noqa: BLE001 - compaction failure is non-fatal
            loop.session.notify("error")
            loop.session.log(f"compaction failed: {e}")
            return False
        finally:
            loop.session.compacting = False
