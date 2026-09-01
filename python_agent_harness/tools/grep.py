"""Grep tool: git grep, then rg, then plain grep.

Grep mirrors `gptel-agent-harness-tools--grep`: git grep (passing the
regex via `-e`), then rg, then plain grep.  Oversized results are
spilled to a temp file (see `filesystem._spool`), so no matches are
ever silently lost.
"""

from __future__ import annotations

import os
import shutil
import subprocess

from .base import Tool, ToolContext
from .filesystem import _git_root, _spool


class Grep(Tool):
    name = "Grep"
    description = (
        "Search file contents with a regular expression. "
        "Use this for content search; use Glob for filename search. "
        "Oversized results are spilled to a temp file (see the 'Stored in:' "
        "path); use Read to view the full output."
    )
    parameters = {
        "type": "object",
        "properties": {
            "regex": {"type": "string", "description": "Regular expression to search for"},
            "path": {"type": "string", "description": "File or directory to search in"},
            "glob": {"type": "string", "description": "Optional file pattern filter (e.g. *.py)"},
            "context_lines": {
                "type": "integer",
                "description": "Lines of context (0-15)",
                "maximum": 15,
            },
        },
        "required": ["regex", "path"],
    }

    def run(self, args: dict, ctx: ToolContext) -> str:
        regex = args["regex"]
        path = os.path.realpath(args["path"])
        if not os.path.isdir(path) and not os.path.isfile(path):
            return f"Error: path {args['path']} is not readable"
        glob = args.get("glob")
        context = args.get("context_lines")
        if context is not None:
            context = max(0, min(15, int(context)))

        git_root = _git_root(path)
        if git_root:
            rel = os.path.relpath(path, git_root)
            pathspec = rel
            if glob and os.path.isdir(path):
                pathspec = os.path.join(rel, glob).replace(os.sep, "/")
            cmd = [
                "git",
                "grep",
                "--line-number",
                "--no-color",
                "--max-count=1000",
                "--untracked",
                "-P",
                "-e",
                regex,
                "--",
                pathspec,
            ]
            if context:
                cmd = cmd[:3] + [f"-C{context}"] + cmd[3:]
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=git_root,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=60,
                )
            except (OSError, subprocess.TimeoutExpired):
                proc = None
            if proc is not None and proc.returncode in (0, 1):
                return _grep_out(proc, "git")
        return self._fallback_rg_grep(regex, path, glob, context)

    def _fallback_rg_grep(
        self, regex: str, path: str, glob: str | None, context: int | None
    ) -> str:
        """rg → grep fallback chain (no git grep).

        Shared by :class:`Grep` (after git grep -P fails) and
        :class:`GrepMac` (after git grep -E fails).  Extracted here so
        the Mac variant can skip the parent's ``git grep -P`` attempt
        without duplicating the rg/grep logic.
        """
        if shutil.which("rg"):
            cmd = [
                "rg",
                "--sort=modified",
                "--max-count=1000",
                "--heading",
                "--line-number",
                "-e",
                regex,
                path,
            ]
            if context:
                cmd = cmd[:1] + [f"--context={context}"] + cmd[1:]
            if glob:
                cmd = cmd[:1] + [f"--glob={glob}"] + cmd[1:]
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=60,
                )
            except (OSError, subprocess.TimeoutExpired):
                proc = None
            if proc is not None and proc.returncode in (0, 1):
                return _grep_out(proc, "rg")
        if shutil.which("grep"):
            cmd = [
                "grep",
                "--recursive",
                "--max-count=1000",
                "--line-number",
                "--regexp",
                regex,
                path,
            ]
            if context:
                cmd = cmd[:1] + [f"--context={context}"] + cmd[1:]
            if glob:
                cmd = cmd[:1] + [f"--include={glob}"] + cmd[1:]
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=60,
                )
            except (OSError, subprocess.TimeoutExpired):
                proc = None
            if proc is not None:
                return _grep_out(proc, "grep")
        return "Error: ripgrep/grep/git-grep not available, this tool cannot be used"


def _grep_out(proc: subprocess.CompletedProcess, backend: str) -> str:
    text = proc.stdout
    if proc.returncode >= 2:
        text = f"Error: search failed with exit-code {proc.returncode}.  Tool output:\n\n{text}"
    return _spool(text, "grep")
