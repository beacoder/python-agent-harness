"""Filesystem tools: Read, Write, Edit, Insert, Mkdir, Glob, Grep.

Glob is git-aware (git ls-files, .gitignore-respecting) with a tree
fallback; Grep prefers rg, then git grep, then plain grep — mirroring
gptel-agent-harness-tools.el.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .base import Tool, ToolContext
from ..diffrender import unified_diff

MAX_OUTPUT = 200_000  # truncation cap (chars)


def _truncate(text: str, label: str = "output") -> str:
    if len(text) > MAX_OUTPUT:
        return text[:MAX_OUTPUT] + f"\n... [truncated {label}]"
    return text


class Read(Tool):
    name = "Read"
    description = (
        "Read the contents of a file. Reads whole file by default, "
        "or a specific line range (start_line/end_line, both inclusive)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path of the file to read"},
            "start_line": {"type": "integer", "description": "First line to read"},
            "end_line": {"type": "integer", "description": "Last line to read"},
        },
        "required": ["file_path"],
    }

    def run(self, args: dict, ctx: ToolContext) -> str:
        path = args["file_path"]
        ctx.guard_path(path, "Read")
        full = os.path.abspath(path)
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except OSError as e:
            return f"Error: cannot read {path}: {e}"
        start = args.get("start_line")
        end = args.get("end_line")
        if start is None and end is None:
            return _truncate("".join(lines), "read")
        start = int(start or 1)
        end = int(end) if end is not None else len(lines)
        if start < 1:
            start = 1
        if end > len(lines):
            end = len(lines)
        if start > end:
            return f"Error: start_line {start} > end_line {end}"
        sel = lines[start - 1:end]
        text = "".join(sel)
        prefix = f"Showing lines {start}-{end} of {len(lines)}:\n\n"
        return prefix + _truncate(text, "read")


class GlobTool(Tool):
    name = "Glob"
    description = (
        "Find files by glob pattern (e.g. '*.py' or 'src/**/test*.py'). "
        "Returns absolute paths. Git-aware: respects .gitignore."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern, e.g. *.py"},
            "path": {"type": "string", "description": "Directory to search in (default: cwd)"},
            "depth": {"type": "integer", "description": "Maximum directory depth (0 or omitted = no limit)"},
        },
        "required": ["pattern"],
    }

    def run(self, args: dict, ctx: ToolContext) -> str:
        pattern = args["pattern"]
        if not pattern:
            return "Error: pattern must not be empty"
        base = os.path.abspath(args.get("path") or ctx.cwd)
        if not os.path.isdir(base):
            return f"Error: path {args.get('path') or ctx.cwd} is not readable"
        ctx.guard_path(base, "Glob")
        depth = args.get("depth")
        if depth is not None:
            depth = int(depth)

        git_root = _git_root(base)
        git_err = None
        if git_root:
            rel = os.path.relpath(base, git_root)
            pathspec = pattern if rel == "." else os.path.join(rel, pattern)
            pathspec = pathspec.replace(os.sep, "/")
            try:
                proc = subprocess.run(
                    ["git", "ls-files", "-z", "--full-name", "--cached",
                     "--others", "--exclude-standard", "--", pathspec],
                    cwd=git_root, capture_output=True, text=True,
                    encoding="utf-8", errors="replace", timeout=60,
                )
            except (OSError, subprocess.TimeoutExpired) as e:
                proc = None
                git_err = str(e)
            else:
                git_err = None
            if proc is not None:
                out = proc.stdout
                if proc.returncode != 0:
                    out += f"Glob failed with exit code {proc.returncode}\n.STDOUT:\n\n"
                return _git_glob_results(out, git_root, base, depth, pattern)

        if git_err and "No such file" in git_err:
            pass  # fall through to tree / python glob
        if shutil.which("tree"):
            cmd = ["tree", "-l", "-f", "-i", "-I", ".git",
                   "--sort=mtime", "--ignore-case", "--prune", "-P", pattern, base]
            if depth is not None and depth > 0:
                cmd += ["-L", str(depth)]
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True,
                    encoding="utf-8", errors="replace", timeout=60,
                )
            except (OSError, subprocess.TimeoutExpired):
                proc = None
            if proc is not None and proc.returncode == 0:
                return _truncate(proc.stdout, "glob")
            if proc is not None and proc.returncode != 0:
                return _truncate(
                    proc.stdout + f"Glob failed with exit code {proc.returncode}\n.STDOUT:\n\n",
                    "glob",
                )
        return "Error: Executable `tree` not found.  This tool cannot be used"


def _git_root(path: str) -> str | None:
    d = Path(path).resolve()
    for parent in [d, *d.parents]:
        if (parent / ".git").exists() or (parent / ".git").is_dir():
            return str(parent)
    return None


def _git_glob_results(
    raw: str, git_root: str, base: str, depth: int | None, pattern: str
) -> str:
    lines = [l for l in raw.split("\0") if l]
    # depth <= 0 means "no limit" (matches `tree -L 0`), so an explicit
    # 0 never produces a confusingly empty result
    if depth is not None and depth > 0:
        base_depth = 0
        rel_base = os.path.relpath(base, git_root)
        if rel_base != ".":
            base_depth = 1 + rel_base.count(os.sep)
        filtered = []
        for l in lines:
            if l.count("/") >= base_depth + depth:
                continue
            filtered.append(l)
        lines = filtered
    out = "\n".join(os.path.join(git_root, l) for l in lines)
    if not out:
        return ""
    return _truncate(out + "\n", "glob")


class Grep(Tool):
    name = "Grep"
    description = (
        "Search file contents with a regular expression. "
        "Use this for content search; use Glob for filename search."
    )
    parameters = {
        "type": "object",
        "properties": {
            "regex": {"type": "string", "description": "Regular expression to search for"},
            "path": {"type": "string", "description": "File or directory to search in"},
            "glob": {"type": "string", "description": "Optional file pattern filter (e.g. *.py)"},
            "context_lines": {"type": "integer", "description": "Lines of context (0-15)"},
        },
        "required": ["regex", "path"],
    }

    def run(self, args: dict, ctx: ToolContext) -> str:
        regex = args["regex"]
        path = os.path.abspath(args["path"])
        if not os.path.isdir(path) and not os.path.isfile(path):
            return f"Error: path {args['path']} is not readable"
        ctx.guard_path(path, "Grep")
        glob = args.get("glob")
        context = args.get("context_lines")
        if context is not None:
            context = max(0, min(15, int(context)))

        git_root = _git_root(path) if os.path.isdir(path) else None
        if git_root:
            rel = os.path.relpath(path, git_root)
            pathspec = rel
            if glob and os.path.isdir(path):
                pathspec = os.path.join(rel, glob).replace(os.sep, "/")
            cmd = ["git", "grep", "--line-number", "--no-color", "--max-count=1000",
                   "--untracked", "-P", "-e", regex, "--", pathspec]
            if context:
                cmd = cmd[:3] + [f"-C{context}"] + cmd[3:]
            try:
                proc = subprocess.run(
                    cmd, cwd=git_root, capture_output=True, text=True,
                    encoding="utf-8", errors="replace", timeout=60,
                )
            except (OSError, subprocess.TimeoutExpired):
                proc = None
            if proc is not None and proc.returncode in (0, 1):
                return _grep_out(proc, "git")
        if shutil.which("rg"):
            cmd = ["rg", "--sort=modified", "--max-count=1000",
                   "--heading", "--line-number", "-e", regex, path]
            if context:
                cmd = cmd[:1] + [f"--context={context}"] + cmd[1:]
            if glob:
                cmd = cmd[:1] + [f"--glob={glob}"] + cmd[1:]
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True,
                    encoding="utf-8", errors="replace", timeout=60,
                )
            except (OSError, subprocess.TimeoutExpired):
                proc = None
            if proc is not None and proc.returncode in (0, 1):
                return _grep_out(proc, "rg")
        if shutil.which("grep"):
            cmd = ["grep", "--recursive", "--max-count=1000",
                   "--line-number", "--regexp", regex, path]
            if context:
                cmd = cmd[:1] + [f"--context={context}"] + cmd[1:]
            if glob:
                cmd = cmd[:1] + [f"--include={glob}"] + cmd[1:]
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True,
                    encoding="utf-8", errors="replace", timeout=60,
                )
            except (OSError, subprocess.TimeoutExpired):
                proc = None
            if proc is not None:
                return _grep_out(proc, "grep")
        return "Error: ripgrep/grep/git-grep not available, this tool cannot be used"


def _grep_out(proc: subprocess.CompletedProcess, backend: str) -> str:
    text = proc.stdout
    if proc.returncode >= 2:
        text = (
            f"Error: search failed with exit-code {proc.returncode}.  "
            f"Tool output:\n\n{text}"
        )
    return _truncate(text, "grep")


class Mkdir(Tool):
    name = "Mkdir"
    description = "Create a new directory (including parents)."
    parameters = {
        "type": "object",
        "properties": {
            "parent": {"type": "string", "description": "Parent directory"},
            "name": {"type": "string", "description": "Directory name to create"},
        },
        "required": ["parent", "name"],
    }

    def run(self, args: dict, ctx: ToolContext) -> str:
        parent = args["parent"]
        name = args["name"]
        path = os.path.abspath(os.path.join(parent, name))
        ctx.guard_path(path, "Mkdir")
        try:
            os.makedirs(path, exist_ok=True)
            return f"Directory {name} created/verified in {parent}"
        except OSError as e:
            return f"Error: {e}"


class Write(Tool):
    name = "Write"
    description = (
        "Create a new file with the given content. "
        "Overwrites an existing file — use with care!"
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory for the file"},
            "filename": {"type": "string", "description": "File name"},
            "content": {"type": "string", "description": "Full file content"},
        },
        "required": ["path", "filename", "content"],
    }

    def run(self, args: dict, ctx: ToolContext) -> str:
        path = os.path.abspath(os.path.join(args["path"], args["filename"]))
        ctx.guard_path(path, "Write")
        existed = os.path.exists(path)
        old_content = ""
        if existed:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    old_content = f.read()
            except OSError:
                old_content = ""
        try:
            ctx.snapshot(path, "Write")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(args["content"])
        except OSError as e:
            return f"Error: {e}"
        if not existed:
            ctx.record_absent(path, "Write")
        diff_text = unified_diff(old_content, args["content"], path)
        if diff_text:
            ctx.record_diff(diff_text)
        return f"Created file {args['filename']} in {args['path']}"


class Edit(Tool):
    name = "Edit"
    description = (
        "Replace text in an existing file. "
        "old_str must exactly match one unique section of the file. "
        "Alternatively provide a unified diff via the diff parameter."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path"},
            "old_str": {"type": "string", "description": "Exact text to replace"},
            "new_str": {"type": "string", "description": "Replacement text"},
            "diff": {"type": "boolean", "description": "Whether new_str is a unified diff"},
        },
        "required": ["path", "new_str"],
    }

    def run(self, args: dict, ctx: ToolContext) -> str:
        path = os.path.abspath(args["path"])
        ctx.guard_path(path, "Edit")
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError as e:
            return f"Error: cannot read {path}: {e}"
        ctx.snapshot(path, "Edit")
        try:
            if args.get("diff"):
                new = _apply_diff(content, args["new_str"])
            else:
                old = args.get("old_str")
                if old is None:
                    return "Error: old_str is required for non-diff edits"
                if content.count(old) == 0:
                    return "Error: old_str not found in file"
                if content.count(old) > 1:
                    return "Error: old_str is not unique; provide more context"
                new = content.replace(old, args["new_str"])
        except Exception as e:
            return f"Error: {e}"
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new)
        except OSError as e:
            return f"Error: {e}"
        diff_text = unified_diff(content, new, path)
        if diff_text:
            ctx.record_diff(diff_text)
        return f"Successfully replaced text in {path}"


_HUNK_HEADER_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_len>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_len>\d+))? @@"
)


@dataclass
class _Hunk:
    old_start: int  # 1-based
    old_len: int
    ops: list[tuple[str, str]]  # (' '|'-'|'+', line-content incl. trailing \n)


def _parse_unified_diff(diff: str) -> list[_Hunk]:
    """Parse a unified diff into hunks; raises ValueError on malformed input."""
    hunks: list[_Hunk] = []
    current: _Hunk | None = None
    for raw in diff.splitlines(keepends=True):
        if raw.startswith("--- ") or raw.startswith("+++ ") or raw.startswith("diff --git"):
            continue
        m = _HUNK_HEADER_RE.match(raw)
        if m:
            if current is not None:
                hunks.append(current)
            old_start = int(m.group("old_start"))
            old_len = int(m.group("old_len") or "1")
            current = _Hunk(old_start=old_start, old_len=old_len, ops=[])
            continue
        if current is None:
            continue  # ignore stray lines before the first hunk header
        if raw.startswith("\\"):
            # "\ No newline at end of file" marker: the preceding
            # content line has no trailing newline.  When the diff is
            # echoed back by the model, that line carries a newline in
            # the text (line separators), so strip it for the strict
            # source-line comparison below.
            if current.ops and current.ops[-1][1].endswith("\n"):
                op, text = current.ops[-1]
                current.ops[-1] = (op, text[:-1])
            continue
        if raw.startswith("+"):
            current.ops.append(("+", raw[1:]))
        elif raw.startswith("-"):
            current.ops.append(("-", raw[1:]))
        elif raw.startswith(" "):
            current.ops.append((" ", raw[1:]))
        elif raw.strip() == "" or raw == "\n":
            current.ops.append((" ", raw))
        else:
            raise ValueError(f"malformed diff line: {raw!r}")
    if current is not None:
        hunks.append(current)
    if not hunks:
        raise ValueError("no hunks found in diff")
    return hunks


def _apply_diff(content: str, diff: str) -> str:
    """Apply a unified diff to CONTENT; raises ValueError on failure.

    Hunks are applied in order using their declared old-file line
    positions; context/removed lines are verified against the source
    so a stale or mismatched hunk fails loudly instead of silently
    corrupting the file.
    """
    hunks = _parse_unified_diff(diff)
    src_lines = content.splitlines(keepends=True)
    result: list[str] = []
    cursor = 0  # 0-based index into src_lines already consumed

    for hunk in hunks:
        start = hunk.old_start - 1 if hunk.old_start > 0 else 0
        if start < cursor:
            raise ValueError("hunks overlap or are out of order")
        # copy untouched lines before this hunk verbatim
        result.extend(src_lines[cursor:start])
        cursor = start
        for op, text in hunk.ops:
            if op == " ":
                if cursor >= len(src_lines) or src_lines[cursor] != text:
                    raise ValueError(
                        f"context line mismatch at line {cursor + 1}: "
                        f"expected {text!r}, found "
                        f"{src_lines[cursor] if cursor < len(src_lines) else '<eof>'!r}"
                    )
                result.append(text)
                cursor += 1
            elif op == "-":
                if cursor >= len(src_lines) or src_lines[cursor] != text:
                    raise ValueError(
                        f"removed line mismatch at line {cursor + 1}: "
                        f"expected {text!r}, found "
                        f"{src_lines[cursor] if cursor < len(src_lines) else '<eof>'!r}"
                    )
                cursor += 1
            elif op == "+":
                result.append(text)

    result.extend(src_lines[cursor:])
    return "".join(result)


class Insert(Tool):
    name = "Insert"
    description = (
        "Insert text at a specific line number in an existing file. "
        "line_number 0 = beginning, -1 = end."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path"},
            "line_number": {"type": "integer", "description": "Line after which to insert (0=start, -1=end)"},
            "new_str": {"type": "string", "description": "Text to insert"},
        },
        "required": ["path", "line_number", "new_str"],
    }

    def run(self, args: dict, ctx: ToolContext) -> str:
        path = os.path.abspath(args["path"])
        ctx.guard_path(path, "Insert")
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.read().splitlines(keepends=True)
        except OSError as e:
            return f"Error: cannot read {path}: {e}"
        ctx.snapshot(path, "Insert")
        ln = int(args["line_number"])
        new_str = args["new_str"]
        if not new_str.endswith("\n"):
            new_str += "\n"
        if ln == -1 or ln >= len(lines):
            lines.append(new_str)
        elif ln == 0:
            lines.insert(0, new_str)
        else:
            lines.insert(ln, new_str)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("".join(lines))
        except OSError as e:
            return f"Error: {e}"
        return f"Successfully inserted text at line {ln} in {path}"
