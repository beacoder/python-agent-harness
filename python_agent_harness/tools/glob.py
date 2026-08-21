"""Glob tool: `git ls-files` inside git repos, `tree` outside.

Glob mirrors `gptel-agent-harness-tools--glob`: inside a git
repository it uses `git ls-files` (fast, .gitignore-respecting), and
falls back to the `tree` command outside git.  Oversized results are
spilled to a temp file (see `filesystem._spool`), so no matches are
ever silently lost.
"""

from __future__ import annotations

import os
import shutil
import subprocess

from .base import Tool, ToolContext
from .filesystem import _git_root, _natnump, _spool


class GlobTool(Tool):
    name = "Glob"
    description = (
        "Recursively find files matching a provided glob pattern.\n\n"
        '- Supports glob patterns like "*.md" or "*test*.py".\n'
        "- Inside a git repository, matching respects .gitignore and covers "
        "both tracked and untracked files.\n"
        "- Returns matching file paths (absolute) at all depths.  Limit the "
        "depth of the search by providing the `depth` argument.\n"
        "- When you are doing an open ended search that may require multiple "
        'rounds of globbing and grepping, use the "Agent" tool instead.\n'
        "- Oversized results are spilled to a temp file (see the 'Stored in:' "
        "path); use Read to view the full output."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": (
                    'Glob pattern to match, for example "*.el". Must not be '
                    'empty.\nUse "*" to list all files in a directory.'
                ),
            },
            "path": {
                "type": "string",
                "description": (
                    'Directory to search in.  Supports relative paths and defaults to "."'
                ),
            },
            "depth": {
                "type": "integer",
                "description": (
                    "Limit directory depth of search, 1 or higher. Defaults to no limit."
                ),
            },
        },
        "required": ["pattern"],
    }

    def run(self, args: dict, ctx: ToolContext) -> str:
        # Mirrors `gptel-agent-harness-tools--glob': `git ls-files' inside a
        # git repository (fast, .gitignore-respecting), `tree' as a fallback
        # outside git.
        pattern = args.get("pattern") or ""
        if not pattern:
            return "Error: pattern must not be empty"
        path = args.get("path")
        if path:
            if not (os.path.isdir(path) and os.access(path, os.R_OK)):
                return f"Error: path {path} is not readable"
        else:
            path = ctx.cwd
        base = os.path.abspath(path)  # directory-file-name + expand-file-name
        depth = args.get("depth")

        git_root = _git_root(base)
        if not git_root and not shutil.which("tree"):
            return "Error: Executable `tree` not found.  This tool cannot be used"

        if git_root:
            rel = os.path.relpath(base, git_root)
            pathspec = pattern if rel == "." else f"{rel}/{pattern}".replace(os.sep, "/")
            try:
                proc = subprocess.run(
                    [
                        "git",
                        "ls-files",
                        "-z",
                        "--full-name",
                        "--cached",
                        "--others",
                        "--exclude-standard",
                        "--",
                        pathspec,
                    ],
                    cwd=git_root,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=60,
                )
            except (OSError, subprocess.TimeoutExpired) as e:
                return f"Error: {e}"
            if proc.returncode != 0:
                # Failure banner is prepended to whatever git emitted.
                banner = f"Glob failed with exit code {proc.returncode}\n.STDOUT:\n\n"
                return _spool(banner + (proc.stdout or "") + (proc.stderr or ""), "glob")
            return _git_glob_results(proc.stdout, git_root, base, depth)

        # --- Tree strategy (fallback outside git) ---
        cmd = [
            "tree",
            "-l",
            "-f",
            "-i",
            "-I",
            ".git",
            "--sort=mtime",
            "--ignore-case",
            "--prune",
            "-P",
            pattern,
            base,
        ]
        if _natnump(depth):
            cmd += ["-L", str(depth)]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            return f"Error: {e}"
        out = proc.stdout
        if proc.returncode != 0:
            out = f"Glob failed with exit code {proc.returncode}\n.STDOUT:\n\n" + out
        return _spool(out, "glob")


def _git_glob_results(raw: str, git_root: str, base: str, depth: object) -> str:
    """Format `git ls-files -z` output into absolute paths, depth-filtered.

    Mirrors the git branch of `gptel-agent-harness-tools--glob': split on
    NUL, drop entries whose slash-count reaches ``base_depth + depth``
    (only when DEPTH is a non-negative integer — `natnump'), then prefix
    each remaining entry with GIT-ROOT.
    """
    lines = [line for line in raw.split("\0") if line]
    if _natnump(depth):
        rel_base = os.path.relpath(base, git_root)
        base_depth = 0 if rel_base == "." else 1 + rel_base.count("/")
        lines = [line for line in lines if line.count("/") < base_depth + depth]
    out = "\n".join(os.path.join(git_root, line) for line in lines)
    if not out:
        return ""
    return _spool(out + "\n", "glob")
