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
_FC_DEC_LINE = r"[ \t]*[*_#>`\-]+[ \t]*"
_FC_DANGLING_RE = re.compile(r"(?:^|\n)" + _FC_DEC_LINE + r"$")
# the same line, matched against a slice range via
# fullmatch(text, pos, endpos) so no substring has to be built
_FC_DEC_LINE_RE = re.compile(_FC_DEC_LINE)

# a complete header on its own, used to spot a block that has started but
# whose Goal/Status/Evidence labels have not all streamed in yet
_FC_HEADER_RE = re.compile(_FC_HEADER, re.IGNORECASE)


def _prefix_of(word: str) -> str:
    """Regex source matching any prefix of WORD, including the empty one.

    ``_prefix_of("ab")`` -> ``(?:a(?:b)?)?``.  Used to recognize a header
    that is still mid-flight in a streamed buffer ("[FINAL CH").
    """
    out = ""
    for ch in reversed(word):
        out = f"(?:{re.escape(ch)}{out})?"
    return out


# A header still being streamed, anchored to the end of the buffer: the
# text from here on may yet turn into a [FINAL CHECK] block, so an
# append-only stream must not emit it (it could never take it back).
#
# Each alternative demands at least one committing character — a "[", a
# decoration run, or an "f" — so ordinary prose and a bare trailing
# newline are never withheld.  The decoration runs are written as
# "one char then a mixed class" rather than a nested quantifier, which
# would risk catastrophic backtracking on a long "****..." tail.
_FC_DEC = r"[*_#>`\-][*_#>`\- \t]*"  # "> ", "**", "###### ", "> **"
_FC_FIN = _prefix_of("final")
_FC_CHK_TAIL = r"(?:[ \t_]*" + _prefix_of("check") + r"[ \t]*\]?)?"
_FC_PARTIAL_TAIL_RE = re.compile(
    "(?:"
    # "[FINAL CH", "**[final" — the bracket commits the match; a
    # bracketed header may sit mid-line, so this is not line-anchored
    + r"(?:\*\*|__|#{1,6}[ \t]*)?\[[ \t]*"
    + _FC_FIN
    + _FC_CHK_TAIL
    # "\n> **", "\n## Final Ch" — a decoration run commits the match.
    # Held because _FC_DANGLING_RE deletes such a run once the block is
    # cut, so emitting it would leak a fragment.
    + r"|(?:^|\n)[ \t]*"
    + _FC_DEC
    + r"(?:\[[ \t]*)?"
    + _FC_FIN
    + _FC_CHK_TAIL
    # "\nFinal Che" — the "f" commits the match
    + r"|(?:^|\n)[ \t]*(?:"
    + _FC_DEC
    + r")?f"
    + _prefix_of("inal")
    + r"(?:[ \t_]+"
    + _prefix_of("check")
    + r")?"
    # "s**" — _FC_HEADER accepts "**"/"__"/"#" immediately before a
    # bracketed header, and that header need not start a line, so a
    # trailing decoration run is ambiguous even mid-sentence.
    + r"|[*_#]{1,6}[ \t]*"
    + r")\Z",
    re.IGNORECASE,
)


# How far back a header-in-flight can start.  The longest one is a
# decorated bracketed header ("###### **[ final _ check ]") plus indent,
# comfortably under this, so a window this size cannot miss one.
_FC_TAIL_WINDOW = 64

# Cheap presence gates.  These replace a "final" in text.lower() test:
# the same O(n) scan in C, but without copying the buffer -- this runs
# once per streamed chunk, so an allocation here costs O(n^2) overall.
_FC_FINAL_WORD_RE = re.compile("final", re.IGNORECASE)
_FC_CHECK_WORD_RE = re.compile("check", re.IGNORECASE)

# A dangling decoration line is a few characters ("> ", "**", "---").
# Bounding the lookback keeps the search off the whole buffer; a
# decoration-only line longer than this is not cleaned up (nor plausible).
_FC_DANGLING_LOOKBACK = 4096


def _is_fc_candidate(text: str, pos: int = 0) -> bool:
    """Whether a complete header could exist at or after POS (cheap gate).

    The header carries both words, so a miss rules out a block without
    running the expensive pattern.
    """
    return (
        _FC_FINAL_WORD_RE.search(text, pos) is not None
        and _FC_CHECK_WORD_RE.search(text, pos) is not None
    )


def _find_final_check(text: str, pos: int = 0) -> re.Match[str] | None:
    """The complete block at or after POS, or None.

    POS lets a streaming caller skip a prefix it has already proven
    block-free, which is what keeps per-chunk work proportional to the
    new text rather than to the whole buffer.  Note that ``^`` cannot
    match at POS > 0 (Python anchors it to the real string start), which
    is correct here: a line-anchored header needs a preceding newline,
    and a caller's proven prefix never ends in whitespace.
    """
    if not _is_fc_candidate(text, pos):
        return None
    return _FINAL_CHECK_RE.search(text, pos)


def _rstrip_end(text: str, end: int) -> int:
    """Index of the end of TEXT[:END] with trailing whitespace dropped."""
    while end > 0 and text[end - 1].isspace():
        end -= 1
    return end


def _cut_end(text: str, start: int) -> int:
    """Length of the kept head when a block is cut at START.

    Mirrors ``head.rstrip()`` plus the ``_FC_DANGLING_RE`` cleanup, but
    works in indices so no intermediate copy of the buffer is made.
    """
    end = _rstrip_end(text, start)
    lo = max(0, end - _FC_DANGLING_LOOKBACK)
    nl = text.rfind("\n", lo, end)
    if nl != -1:
        line_start = nl + 1
    elif lo == 0:
        line_start = 0  # the head is a single line
    else:
        return end  # last line too long to be a decoration run
    if line_start < end and _FC_DEC_LINE_RE.fullmatch(text, line_start, end):
        # drop the decoration-only line, then its preceding whitespace
        return _rstrip_end(text, nl if nl != -1 else 0)
    return end


def final_check_hold_index(text: str, floor: int = 0) -> int:
    """Length of the prefix of TEXT an append-only stream may emit.

    ``strip_final_check`` is not monotonic: when a [FINAL CHECK] block
    completes, the visible text SHRINKS (the block, and any whitespace
    before it, disappear).  A stream of appended chunks cannot retract
    what it already sent, so it has to withhold every byte that a later
    chunk might delete.  That is:

    - everything from a header that has already matched but whose block
      is still incomplete (the labels may still arrive),
    - everything from a header that is itself still mid-flight,
    - trailing whitespace, which ``strip_final_check`` rstrips away
      when it cuts a block.

    The withheld tail is not lost — it becomes emittable as soon as the
    following chunk rules the block out, and ``FilteredDeltaStream.flush``
    releases whatever is left at the end of a message.

    Withholding too much would only delay text (flush still releases
    it); withholding too little would leak a block fragment, so the
    checks below err toward holding.  ``floor`` is a prefix the caller
    has already emitted -- and therefore already proven safe -- so the
    searches may skip it.
    """
    hold = len(text)
    m = _FC_HEADER_RE.search(text, floor) if _is_fc_candidate(text, floor) else None
    if m is not None:
        hold = min(hold, m.start())
    # A header still in flight ("\n\n**", "[FINAL CH") need not contain
    # either word yet, so that gate cannot apply here.  This search is
    # bounded to the tail instead: a header prefix is far shorter than
    # the window, so nothing can start before it.
    partial = _FC_PARTIAL_TAIL_RE.search(text, max(floor, len(text) - _FC_TAIL_WINDOW))
    if partial is not None:
        hold = min(hold, partial.start())
    # strip_final_check also deletes a trailing decoration-only line (the
    # "> " or "**" left in front of a cut header), and that line can sit
    # before the hold point -- e.g. "a\n\n> \n[FINAL CHECK]..." collapses
    # to "a".  Withhold it too, for the same reason.
    return _cut_end(text, hold)


def strip_final_check(text: str, floor: int = 0) -> str:
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

    ``floor`` skips a prefix the caller has already proven block-free
    (a streaming caller's emitted text); it defaults to scanning all of
    TEXT.
    """
    # The searches are expensive on large buffers -- the live stream row
    # can be ~100K chars and is re-stripped every frame -- so they run
    # behind a cheap presence gate and never copy the buffer.
    m = _find_final_check(text, floor)
    if m is None:
        return text
    return text[: _cut_end(text, m.start())]


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
