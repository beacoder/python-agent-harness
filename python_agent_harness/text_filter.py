"""Text filters shared by the TUI and the headless view.

These trim verification bookkeeping out of assistant replies for
display/output.  The stored messages are never modified.
"""

from __future__ import annotations

import re

# the completion-check filter: a FINAL CHECK header followed by the
# Goal/Status/Evidence labels (anywhere in the block, any lines).
#
# Models reformat the block from task-completion-rules.md freely, so the
# pattern must tolerate markdown decoration.  Seen in the wild:
# "[FINAL CHECK]", "**[FINAL CHECK]**", "## Final Check", and labels as
# "Goal:", "**Goal:**" or "**Goal**:" (colon outside the emphasis) —
# the last variant has no literal "Goal:" in it, which is what made the
# old literal pattern miss and leak the block into the panel.
#
# The header must be bracketed or start its own line: that keeps prose
# like "let me do the final check" from truncating a real reply.
_FC_LABEL = r"[*_`]*[ \t]*:"  # "Goal:", "**Goal:**", "**Goal**:", "`Goal` :"
_FC_HEADER = (
    r"(?:"
    r"(?:\*\*|__|#{1,6}[ \t]*)?"  # decoration before a bracketed header
    r"\[[ \t]*final[ \t_]*check[ \t]*\]"  # [FINAL CHECK], bracketed anywhere
    r"|(?:^|\n)[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*|__)?[ \t]*"
    r"final[ \t_]+check\b"  # ## Final Check / **FINAL CHECK**, line-anchored
    r")"
)
_FINAL_CHECK_RE = re.compile(
    _FC_HEADER + rf".*?Goal{_FC_LABEL}.*?Status{_FC_LABEL}.*?Evidence{_FC_LABEL}",
    re.DOTALL | re.IGNORECASE,
)
# a line left holding nothing but markdown decoration once the block is
# cut away (e.g. the "> " or "**" in front of a decorated header)
_FC_DANGLING_RE = re.compile(r"(?:^|\n)[ \t]*[*_#>`\-]+[ \t]*$")


def strip_final_check(text: str) -> str:
    """Drop a completion-check block from an assistant reply.

    The task-completion rules make the model end with a [FINAL CHECK]
    block (Goal / Status / Evidence) — verification bookkeeping, not
    content the user wants to read.  The filter is ``_FINAL_CHECK_RE``
    (header + the three labels, markdown decoration tolerated):
    everything from the header onward is dropped.

    The block is hidden even when it is the reply's ONLY content —
    check-only replies never render.  Replies without the header are
    untouched.  The agent loop still produces and stores the message
    unchanged; this only trims it from the display/output.
    """
    # Fast path: the regex below is expensive on large buffers (the live
    # stream row can be ~100K chars and is re-stripped every frame), so
    # gate it behind a cheap substring check.  The header always contains
    # both words, so a miss means no block to strip.
    low = text.lower()
    if "final" not in low or "check" not in low:
        return text
    m = _FINAL_CHECK_RE.search(text)
    if m is not None:
        head = text[: m.start()].rstrip()
        return _FC_DANGLING_RE.sub("", head).rstrip()
    return text


def strip_reasoning(text: str, reasoning: str) -> str:
    """Remove the leading REASONING block from TEXT, or TEXT unchanged.

    Reasoning content is streamed before the answer, so it forms the
    leading part of the stored message content.  The TUI collapses it
    to a marker once the stream is done, so it stops eating the
    visible-row budget; the stored message is never modified.
    """
    if not reasoning:
        return text
    if text.startswith(reasoning):
        return text[len(reasoning) :]
    stripped = text.lstrip()
    if stripped.startswith(reasoning):
        return stripped[len(reasoning) :]
    return text
