"""Edit tool: string replacement and diff/patch modes.

Edit mirrors `gptel-agent--edit-files` (inherited unchanged by the
harness): a string-replacement mode (exact unique `old_str` → `new_str`,
single files only) and a diff/patch mode that shells out to
`patch --forward` and works on both single files and whole directories
(multi-file unified diffs), with ```diff code-fence removal and
hunk-header line-count fixing.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess

from ..diffrender import unified_diff
from .base import Tool, ToolContext


class Edit(Tool):
    name = "Edit"
    description = (
        "Replace text in one or more files.\n\n"
        "To edit a single file, provide the file `path`.\n\n"
        "For the replacement, there are two methods:\n"
        "- Short replacements: Provide both `old_str` and `new_str`, in which "
        "case `old_str` needs to exactly match one unique section of the "
        "original file, including any whitespace.  Make sure to include "
        "enough context that the match is not ambiguous.  The entire original "
        "string will be replaced with `new_str`.\n"
        "- Long or involved replacements: set the `diff` parameter to true and "
        "provide a unified diff in `new_str`. `old_str` can be ignored.\n\n"
        "To edit multiple files,\n"
        "- provide the directory path,\n"
        "- set the `diff` parameter to true\n"
        "- and provide a unified diff in `new_str`.\n\n"
        "Diff instructions:\n"
        "- The diff must be in unified format (optionally within a ```diff "
        "fenced code block).\n"
        "- The file paths within the diff (e.g. '--- a/filename' "
        "'+++ b/filename') must be appropriate for the `path`.\n\n"
        'To simply insert text at some line, use the "Insert" tool instead.'
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path or directory to edit"},
            "old_str": {
                "type": "string",
                "description": "Original string to replace.  If providing a unified diff, this should be false",
            },
            "new_str": {"type": "string", "description": "Replacement string OR unified diff text"},
            "diff": {
                "type": "boolean",
                "description": "Whether the replacement is a string or a diff.  `true` for a diff, `false` otherwise.",
            },
        },
        "required": ["path", "new_str"],
    }

    def run(self, args: dict, ctx: ToolContext) -> str:
        raw = args["path"]
        path = os.path.realpath(os.path.abspath(raw))
        if not os.access(path, os.R_OK):
            return f"Error: File or directory {path} is not readable"
        new_str = args.get("new_str")
        if new_str is None:
            return "Error: Required argument `new_str' missing"
        old = args.get("old_str")
        diffp = args.get("diff")
        # gptel: string mode when `diff` is false OR `old_str` is provided.
        if diffp is False or old is not None:
            return self._string_replace(path, old, new_str, ctx)
        # Diff mode runs `patch` in Emacs `file-name-directory' of the path:
        # a trailing-slash directory path -> that directory itself, so a
        # multi-file diff applies to files within it; otherwise the parent.
        if raw.endswith("/") or raw.endswith(os.sep):
            cwd = path or "/"
        else:
            cwd = os.path.dirname(path) or "/"
        return self._apply_patch(path, cwd, new_str, ctx)

    def _string_replace(self, path: str, old: str | None, new_str: str, ctx: ToolContext) -> str:
        if os.path.isdir(path):
            return (
                f"Error: String replacement is intended for single files, not directories ({path})"
            )
        if old is None:
            return "Error: old_str is required for non-diff edits"
        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()
        except OSError as e:
            return f"Error: cannot read {path}: {e}"
        count = content.count(old)
        if count == 0:
            return f'Error: Could not find old_str "{old[:20]}" in file {path}'
        if count > 1:
            return (
                "Error: Match is not unique. Consider providing more context "
                "for the replacement, or a unified diff"
            )
        new = content.replace(old, new_str, 1)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new)
        except OSError as e:
            return f"Error: {e}"
        diff_text = unified_diff(content, new, path)
        if diff_text:
            ctx.record_diff(diff_text)
        return f"Successfully replaced {old[:20]} (truncated) with {new_str[:20]} (truncated)"

    def _apply_patch(self, path: str, cwd: str, diff: str, ctx: ToolContext) -> str:
        """Diff/patch mode: shell out to `patch --forward` (files or dirs).

        Mirrors the diff branch of `gptel-agent--edit-files`: ensure a
        trailing newline, strip a ```diff code fence, fix hunk-header line
        counts, then run `patch` in CWD (Emacs `file-name-directory' of the
        path).  For a single file the before/after contents are captured to
        record a diff for the UI; directory (multi-file) patches skip that.
        """
        if not shutil.which("patch"):
            return (
                'Error: Command "patch" not available, cannot apply diffs. '
                "Use string replacement instead"
            )
        text = diff if diff.endswith("\n") else diff + "\n"
        text = _strip_diff_fence(text)
        text = _fix_patch_headers(text)
        is_file = os.path.isfile(path)
        old_content = None
        if is_file:
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    old_content = f.read()
            except OSError:
                old_content = None
        options = ["--forward", "--verbose"]
        try:
            proc = subprocess.run(
                ["patch", *options],
                input=text,
                cwd=cwd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            return f"Error: {e}"
        out = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode != 0:
            return (
                f"Error: Failed to apply diff to {path} (exit status "
                f"{proc.returncode}).\nPatch command options: {options}\n"
                f"Patch STDOUT:\n{out}"
            )
        if old_content is not None and os.path.isfile(path):
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    new_content = f.read()
                diff_text = unified_diff(old_content, new_content, path)
                if diff_text:
                    ctx.record_diff(diff_text)
            except OSError:
                pass
        return (
            f"Diff successfully applied to {path}.\n"
            f"Patch command options: {options}\n"
            f"Patch STDOUT:\n{out}"
        )


_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+),(\d+) +\+(\d+),(\d+) @@")
# A file section's header pair: the marker must be followed by whitespace
# (real headers name a path), which content lines rendered ---/+++ by the
# diff itself normally are not.
_FILE_HEADER_OLD_RE = re.compile(r"^---[ \t]")
_FILE_HEADER_NEW_RE = re.compile(r"^\+\+\+[ \t]")


def _strip_diff_fence(text: str) -> str:
    """Remove a leading ```diff fence and its trailing ``` line.

    Mirrors the fence handling in `gptel-agent--edit-files`: only a
    ```diff opening fence is stripped (a bare ``` or ```patch is left
    for `patch` to reject), together with the closing ``` line.
    """
    lines = text.splitlines(keepends=True)
    if lines and re.match(r"^ *```diff", lines[0]):
        lines = lines[1:]
        if lines and re.match(r"^ *```", lines[-1]):
            lines = lines[:-1]
        return "".join(lines)
    return text


def _fix_patch_headers(diff_text: str) -> str:
    """Recompute the line counts in unified-diff hunk headers.

    Mirrors `gptel-agent--fix-patch-headers`: for every ``@@ -a,b +c,d @@``
    header, recount the body (context/removed/added lines) up to the next
    header or EOF and rewrite ``b`` and ``d`` accordingly, so a model that
    miscounts hunk lengths still produces a patch `patch` will accept.
    Headers without explicit counts are passed through untouched.
    """
    lines = diff_text.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        m = _HUNK_HEADER_RE.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        orig_line, new_line = int(m.group(1)), int(m.group(3))
        j = i + 1
        orig_count = new_count = 0
        body: list[str] = []
        while j < n and not lines[j].startswith("@@"):
            line = lines[j]
            if line.startswith("---") and _starts_file_section(lines, j):
                # A ---/+++ pair introducing the next file; not hunk body.
                break
            if line.startswith("-"):
                orig_count += 1
            elif line.startswith("+"):
                new_count += 1
            elif line.startswith(" "):
                orig_count += 1
                new_count += 1
            body.append(line)
            j += 1
        out.append(f"@@ -{orig_line},{orig_count} +{new_line},{new_count} @@\n")
        out.extend(body)
        i = j
    return "".join(out)


def _starts_file_section(lines: list[str], idx: int) -> bool:
    """Whether lines[idx] begins the next file's ---/+++ header pair.

    A new file section in a multi-file diff is ``--- path`` immediately
    followed by ``+++ path`` and then a hunk header.  Removed/added
    content lines whose text merely starts with ``--``/``++`` (rendered
    ``---``/``+++``) are hunk body and must be counted, so all three
    parts are required:

    * the ordered pair — a lone ``---``-rendered content line as a
      hunk's LAST body line is followed directly by the next hunk
      header, which a peek for a trailing ``@@`` alone cannot tell from
      a file header (it silently dropped the line from the count and
      made `patch` reject the whole diff);
    * the space/tab after the marker — real headers carry a path
      (``--- a/f``), content lines usually do not (``---removed``);
    * the trailing hunk header.

    A hunk whose last two body lines happen to be a removed line
    starting with ``-- `` AND an added line starting with ``++ `` is
    still indistinguishable from a file header pair by shape alone; it
    stays a known (and far rarer) miscount.
    """
    if not _FILE_HEADER_OLD_RE.match(lines[idx]):
        return False
    if idx + 1 >= len(lines) or not _FILE_HEADER_NEW_RE.match(lines[idx + 1]):
        return False
    return idx + 2 < len(lines) and lines[idx + 2].startswith("@@")
