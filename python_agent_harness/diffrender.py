"""Unified-diff generation and rich rendering for file-changing tools.

``unified_diff`` builds a standard unified diff between two file
contents (used by Edit/Write to record what actually changed).
``render_diff`` turns that text into a red/green ``rich`` renderable
suitable for the TUI's tool-output panel.
"""

from __future__ import annotations

import difflib

from rich.console import Group
from rich.text import Text

MAX_DIFF_LINES = 400  # truncation cap for the rendered (not stored) diff


def unified_diff(
    old_content: str,
    new_content: str,
    path: str,
    context_lines: int = 3,
) -> str:
    """Return a unified diff string between OLD_CONTENT and NEW_CONTENT.

    Empty string when the two are identical (nothing to show).
    Lines at EOF without a trailing newline get a git-style
    ``\\ No newline at end of file`` marker so the diff round-trips
    (the Edit tool's diff mode parses markers and applies them).
    """
    if old_content == new_content:
        return ""
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        n=context_lines,
        lineterm="\n",
    )
    out: list[str] = []
    for line in diff:
        out.append(line)
        if line[:1] in ("-", "+", " ") and not line.endswith("\n"):
            # a content line at EOF without a trailing newline: difflib
            # emits it bare, which would corrupt the joined text; mark
            # it like git does so the diff round-trips (the Edit tool's
            # diff mode parses markers and applies them)
            out.append("\n\\ No newline at end of file\n")
    return "".join(out)


def render_diff(diff_text: str, max_lines: int = MAX_DIFF_LINES) -> Group:
    """Render a unified diff as a rich renderable (red '-' / green '+').

    Hunk headers (@@ ...@@) and file headers (---/+++) are dimmed;
    added lines are green, removed lines are red, context lines are
    plain.  Long diffs are truncated with a marker line.
    """
    lines = diff_text.splitlines()
    truncated = len(lines) > max_lines
    if truncated:
        lines = lines[:max_lines]

    rows: list[Text] = []
    for line in lines:
        if line.startswith("+++") or line.startswith("---"):
            rows.append(Text(line, style="dim bold"))
        elif line.startswith("@@"):
            rows.append(Text(line, style="cyan"))
        elif line.startswith("+"):
            rows.append(Text(line, style="green"))
        elif line.startswith("-"):
            rows.append(Text(line, style="red"))
        else:
            rows.append(Text(line, style="dim"))
    if truncated:
        rows.append(Text("… [diff truncated]", style="dim italic"))
    if not rows:
        rows.append(Text("(no changes)", style="dim italic"))
    return Group(*rows)
