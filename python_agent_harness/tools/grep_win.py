"""Windows Grep tool: ``git grep -P`` then ``rg``, then pure-Python ``re``.

Windows ships neither ``grep`` nor ``rg`` (ripgrep).  ``git grep -P``
works when Git for Windows is installed (it includes PCRE support),
and ``rg`` works when the user has installed it separately.  But when
neither external tool is available, the Linux/macOS fallback chain
returns an error.

``GrepWindows`` extends the fallback chain with a pure-Python
``re``-based search: it walks the directory tree with
:meth:`pathlib.Path.rglob`, applies the regex to each file, and
collects matches with line numbers and optional context lines.  This
ensures the Grep tool always works on a stock Windows install with
only Python and Git installed.

Only the fallback strategy differs; the git path (``git grep -P``),
the tool name, and the result format are inherited so callers, the
tool registry, and the plan-mode write guard are platform-independent.
"""

from __future__ import annotations

import fnmatch
import os
import re
import subprocess
from pathlib import Path

from .filesystem import _spool
from .grep import Grep, _grep_out


class GrepWindows(Grep):
    """Grep with a pure-Python ``re`` fallback for Windows."""

    def _fallback_rg_grep(
        self, regex: str, path: str, glob: str | None, context: int | None
    ) -> str:
        """rg → pure-Python ``re`` fallback chain (no ``grep``).

        Tries ``rg`` (ripgrep) first — it may be installed on Windows
        via scoop/choco/winget.  When ``rg`` is not available or fails,
        falls back to a pure-Python ``re``-based search that walks the
        directory tree, applies the regex to each file, and collects
        matches with line numbers and optional context lines.
        """
        import shutil

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

        return self._python_grep(regex, path, glob, context)

    def _python_grep(self, regex: str, path: str, glob: str | None, context: int | None) -> str:
        """Pure-Python regex search: walk files, match lines, collect results.

        Mirrors the output format of ``git grep`` / ``rg``: ``path:line:content``
        with optional context lines.  Respects the ``glob`` filter for
        file pattern matching.  Results are capped at 1000 matches and
        spilled to a temp file via ``_spool`` when oversized.
        """
        try:
            pattern = re.compile(regex)
        except re.error as e:
            return f"Error: invalid regex: {e}"

        results: list[str] = []
        match_count = 0
        max_matches = 1000

        if os.path.isfile(path):
            files = [Path(path)]
        else:
            root = Path(path)
            files = (
                p
                for p in root.rglob("*")
                if p.is_file()
                and not any(part.startswith(".") for part in p.relative_to(root).parts[:-1])
            )

        for file_path in files:
            if match_count >= max_matches:
                break
            if glob and not fnmatch.fnmatch(file_path.name, glob):
                continue
            try:
                with open(file_path, encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
            except OSError:
                continue
            for i, line in enumerate(lines, 1):
                if pattern.search(line):
                    if match_count >= max_matches:
                        break
                    match_count += 1
                    if context and context > 0:
                        start = max(0, i - 1 - context)
                        end = min(len(lines), i + context)
                        for j in range(start, end):
                            marker = ":" if j + 1 != i else ":"
                            results.append(f"{file_path}:{j + 1}{marker}{lines[j].rstrip()}")
                        results.append("")  # blank line between matches with context
                    else:
                        results.append(f"{file_path}:{i}:{line.rstrip()}")

        out = "\n".join(results)
        if not out:
            return ""
        return _spool(out + "\n", "grep")
