"""Filesystem tools: Read, Write, Edit, Insert, Mkdir, Glob, Grep.

Glob mirrors `gptel-agent-harness-tools--glob`: inside a git repository
it uses `git ls-files` (fast, .gitignore-respecting), and falls back to
the `tree` command outside git.  Grep mirrors
`gptel-agent-harness-tools--grep`: git grep (passing the regex via
`-e`), then rg, then plain grep.

Edit mirrors `gptel-agent--edit-files` (inherited unchanged by the
harness): a string-replacement mode (exact unique `old_str` → `new_str`,
single files only) and a diff/patch mode that shells out to
`patch --forward` and works on both single files and whole directories
(multi-file unified diffs), with ```diff code-fence removal and
hunk-header line-count fixing.

NOTE: these filesystem tools are intentionally SYNCHRONOUS.  Only Bash
and Agent are `:async t` in gptel-agent; sync tools run one at a time in
the model-emitted order.  Do NOT be tempted to port every tool to async
for parallelism: tools can depend on one another's side effects within a
single round (e.g. Write/Mkdir then Read/Edit the same path, or Edit then
Grep the just-changed file).  Running them concurrently would introduce
read-after-write races and non-deterministic results.  Keep filesystem
tools synchronous so ordering — and therefore correctness — is preserved.

Oversized Glob/Grep results are spilled to a temp file (mirroring
`gptel-agent--truncate-buffer` in gptel-agent-tools.el): the tool
result then carries a short preview plus the temp-file path, so the
full output remains readable via the Read tool.

Read mirrors `gptel-agent--read-file-lines`: whole-file reads are
refused above READ_SIZE_LIMIT (400 KB, matching
`gptel-agent-read-file-size-threshold`), and line ranges are streamed
instead of loading the whole file into memory.  Oversized range
results are spilled to a temp file like Glob/Grep results, so nothing
is ever silently truncated.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import TypeGuard

from ..diffrender import unified_diff
from .base import Tool, ToolContext

MAX_OUTPUT = 20_000  # spill threshold (chars), matching gptel-agent--truncate-buffer
SPOOL_LINES = 50  # preview lines kept when results are spilled
READ_SIZE_LIMIT = 400 * 1024  # whole-file reads above this are refused
# (mirrors gptel-agent-read-file-size-threshold)

_spooled_files: list[str] = []  # temp files created by _spool, cleaned
# up by cleanup_spooled_files on session close


def _truncate(text: str, label: str = "output") -> str:
    """In-memory truncation fallback (used when spooling to disk fails)."""
    if len(text) > MAX_OUTPUT:
        return text[:MAX_OUTPUT] + f"\n... [truncated {label}]"
    return text


def _spool_dir() -> str:
    """Reliable temp dir for spilled results (first candidate set, else /tmp)."""
    for d in (
        os.environ.get("TMPDIR"),
        os.environ.get("TMP"),
        os.environ.get("TEMP"),
        tempfile.gettempdir(),
    ):
        if d:
            return os.path.abspath(d)
    return "/tmp"


def _spool(text: str, label: str) -> str:
    """Spill oversized tool output to a temp file; return a preview.

    Mirrors `gptel-agent--truncate-buffer': when TEXT exceeds
    MAX_OUTPUT chars the full content is written to a temp file and the
    returned string becomes a header (size + path), the first
    SPOOL_LINES lines, and a footer telling the agent to Read the file.
    Falls back to in-memory truncation if the temp file cannot be
    written."""
    if len(text) <= MAX_OUTPUT:
        return text
    stamp = time.strftime("%Y%m%d-%H%M%S")
    try:
        fd, temp_file = tempfile.mkstemp(
            prefix=f"python-agent-harness-{label}-{stamp}-",
            suffix=".txt",
            dir=_spool_dir(),
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        _spooled_files.append(temp_file)
    except OSError:
        return _truncate(text, label)
    lines = text.splitlines()
    preview = "\n".join(lines[:SPOOL_LINES])
    return (
        f"{label} results too large ({len(text)} chars, {len(lines)} lines) "
        f"for context window.\n"
        f"Stored in: {temp_file}\n\n"
        f"First {SPOOL_LINES} lines:\n\n"
        f"{preview}\n\n"
        f'[Use Read tool with file_path="{temp_file}" to view full results]'
    )


def cleanup_spooled_files() -> None:
    """Delete all tracked spooled temp files (best effort).

    Mirrors ``PlanMode.cleanup_plan_file``: called from
    ``AgentSession.close`` so oversized tool results do not accumulate
    in the temp dir.  Files already removed (e.g. by a restored
    session) are skipped.
    """
    paths = _spooled_files[:]
    _spooled_files.clear()
    for path in paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass


class Read(Tool):
    name = "Read"
    description = (
        "Read file contents between specified line numbers `start_line` and "
        "`end_line`, with both ends included.\n\n"
        'Consider using the "Grep" tool to find the right range to read first.\n\n'
        "Reads the whole file if the line range is not provided.\n\n"
        f"Files over {READ_SIZE_LIMIT // 1024} KB in size can only be read by "
        "specifying a line range.\n"
        "Very large line ranges are spilled to a temp file (see the 'Stored "
        "in:' path); use Read to view the full output."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "The path to the file to be read"},
            "start_line": {
                "type": "integer",
                "description": "The line to start reading from, defaults to the start of the file",
            },
            "end_line": {
                "type": "integer",
                "description": "The line up to which to read, defaults to the end of the file.",
            },
        },
        "required": ["file_path"],
    }

    def run(self, args: dict, ctx: ToolContext) -> str:
        path = args["file_path"]
        full = os.path.realpath(os.path.abspath(path))
        if os.path.isdir(full):
            return f"Error: cannot read {path}: is a directory"
        try:
            size = os.path.getsize(full)
        except OSError as e:
            return f"Error: cannot read {path}: {e}"
        start = args.get("start_line")
        end = args.get("end_line")
        if start is None and end is None:
            if size > READ_SIZE_LIMIT:
                return (
                    f"Error: File is too large ({size // 1024} KB > "
                    f"{READ_SIZE_LIMIT // 1024} KB). Please specify a line "
                    "range to read"
                )
            try:
                with open(full, encoding="utf-8", errors="replace") as f:
                    return f.read()
            except OSError as e:
                return f"Error: cannot read {path}: {e}"
        start = int(start or 1)
        if start < 1:
            start = 1
        if end is not None:
            end = int(end)
            if start > end:
                return f"Error: start_line {start} > end_line {end}"
        # Stream the file line by line instead of loading it whole, so
        # huge files can be read in ranges with constant memory.
        selected: list[str] = []
        total: int | None = None  # exact line count once EOF is reached
        reached_eof = True
        try:
            with open(full, encoding="utf-8", errors="replace") as f:
                for lineno, line in enumerate(f, 1):
                    if end is not None and lineno > end:
                        reached_eof = False
                        break
                    total = lineno
                    if lineno >= start:
                        selected.append(line)
        except OSError as e:
            return f"Error: cannot read {path}: {e}"
        if end is not None and reached_eof and total is not None:
            end = min(end, total)  # clamp to the real file end
        if total is None:
            total = 0  # empty file
        if start > total:
            return f"Error: start_line {start} > end_line {end if end is not None else total}"
        end_eff = end if end is not None else total
        if total is not None and reached_eof:
            header = f"Showing lines {start}-{end_eff} of {total}:\n\n"
        else:
            header = f"Showing lines {start}-{end_eff}:\n\n"
        return _spool(header + "".join(selected), "read")


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


def _natnump(n: object) -> TypeGuard[int]:
    """True for a non-negative integer (Emacs `natnump' semantics)."""
    return isinstance(n, int) and not isinstance(n, bool) and n >= 0


def _git_root(path: str) -> str | None:
    d = Path(path).resolve()
    for parent in [d, *d.parents]:
        # .git is a directory in a normal clone and a file in worktrees
        # / submodules; exists() covers both.
        if (parent / ".git").exists():
            return str(parent)
    return None


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
        path = os.path.abspath(args["path"])
        if not os.path.isdir(path) and not os.path.isfile(path):
            return f"Error: path {args['path']} is not readable"
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
        path = os.path.realpath(os.path.abspath(os.path.join(parent, name)))
        try:
            os.makedirs(path, exist_ok=True)
            return f"Directory {name} created/verified in {parent}"
        except OSError as e:
            return f"Error: {e}"


class Write(Tool):
    name = "Write"
    description = (
        "Create a new file with the given content. Overwrites an existing file — use with care!"
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
        dir_path = args.get("path") or "."
        filename = args.get("filename") or ""
        content = args.get("content") or ""
        # LLM may put the full file path in "filename" or in "path"
        if filename:
            path = os.path.realpath(os.path.abspath(os.path.join(dir_path, filename)))
        else:
            path = os.path.realpath(os.path.abspath(dir_path))
        if not filename:
            filename = os.path.basename(path) or os.path.basename(dir_path)
        existed = os.path.exists(path)
        old_content = ""
        if existed:
            try:
                with open(path, encoding="utf-8") as f:
                    old_content = f.read()
            except OSError:
                old_content = ""
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        except OSError as e:
            return f"Error: {e}"
        diff_text = unified_diff(old_content, content, path)
        if diff_text:
            ctx.record_diff(diff_text)
        return f"Created file {filename} in {dir_path}"


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
            if line.startswith("---") or line.startswith("+++"):
                # File header lines (--- a/file, +++ b/file) are not
                # hunk body lines; pass them through without counting.
                pass
            elif line.startswith("-"):
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
            "line_number": {
                "type": "integer",
                "description": "Line after which to insert (0=start, -1=end)",
            },
            "new_str": {"type": "string", "description": "Text to insert"},
        },
        "required": ["path", "line_number", "new_str"],
    }

    def run(self, args: dict, ctx: ToolContext) -> str:
        path = os.path.realpath(os.path.abspath(args["path"]))
        try:
            with open(path, encoding="utf-8") as f:
                old_content = f.read()
                lines = old_content.splitlines(keepends=True)
        except OSError as e:
            return f"Error: cannot read {path}: {e}"
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
        new_content = "".join(lines)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)
        except OSError as e:
            return f"Error: {e}"
        diff_text = unified_diff(old_content, new_content, path)
        if diff_text:
            ctx.record_diff(diff_text)
        return f"Successfully inserted text at line {ln} in {path}"
