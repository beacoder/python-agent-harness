"""Session persistence: auto-save, LLM titles, restore.

Ported from gptel-agent-harness-session.el.

- Auto-save the conversation after each LLM response to
  ~/.local/share/python-agent-harness/sessions/<name>_<YYMMDDHHMMSS>.md
  with a trailing metadata block (;; Local Variables: ...).
- Async title generation from the first user message (title.txt);
  on success the file is renamed to <title>_<TS>.md.
- restore / restore-latest commands re-load a session.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

from . import config


def session_dir() -> Path:
    return config.SESSION_DIR / config.SESSION_SUBDIR


def sanitize_title(title: str) -> str:
    """Sanitize a generated title (mirrors the elisp semantics)."""
    t = title.strip()
    t = re.sub(r"[\n\r]+", " ", t)
    t = t.strip("\"")
    t = re.sub(r"[/\\:*?\"<>|]", "-", t)
    t = re.sub(r"[-_ ]+", "-", t)
    t = t[:50]
    t = re.sub(r"-+$", "", t)
    return t


def title_from_filename(session_file: str) -> str | None:
    """Derive a title from a session file name, or None."""
    base = os.path.basename(session_file)
    if base.endswith(".md"):
        base = base[:-3]
    m = re.match(r"(.+)_[0-9]{12}(?:-\d+)?$", base)
    if not m:
        return None
    title = m.group(1).replace("-", " ")
    if " " not in title:
        return None  # bare single-word project names rejected
    return title


class SessionStore:
    """Saves/restores sessions for one agent session."""

    def __init__(self, project_dir: str, model: str, backend: str,
                 system_prompt: str | None = None,
                 temperature: float | None = None,
                 max_tokens: int | None = None,
                 tool_names: list[str] | None = None) -> None:
        self.project_dir = project_dir
        self.model = model
        self.backend = backend
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tool_names = tool_names or []
        self.title: str | None = None
        self.file_path: str | None = None
        self.title_pending = False
        self._first_user_msg: str | None = None

    # -- file naming ----------------------------------------------------------
    @staticmethod
    def _unique_path(prefix: str) -> str:
        """A session file path that does not already exist.

        Same-second collisions (same project name, or two sessions given
        the same title) get a numeric suffix instead of silently
        overwriting an existing session file.
        """
        path = session_dir() / f"{prefix}.md"
        n = 1
        while path.exists():
            path = session_dir() / f"{prefix}-{n}.md"
            n += 1
        return str(path)

    def session_file(self) -> str | None:
        if self.file_path:
            return self.file_path
        proj_name = os.path.basename(os.path.normpath(self.project_dir))
        stamp = time.strftime("%y%m%d%H%M%S")
        self.file_path = self._unique_path(f"{proj_name}_{stamp}")
        return self.file_path

    def remember_first_user_message(self, text: str) -> None:
        if self._first_user_msg is None and len(text.strip()) > 3:
            self._first_user_msg = text.strip()[:500]

    def first_user_message(self) -> str | None:
        return self._first_user_msg

    # -- saving ---------------------------------------------------------------
    def metadata_block(self) -> str:
        lines = [";; Local Variables:"]
        pairs = [
            ("python-agent-harness--project-dir", self.project_dir),
            ("gptel-model", self.model),
            ("gptel--backend-name", self.backend),
            ("gptel-system-prompt", self.system_prompt),
            ("gptel-temperature", self.temperature),
            ("gptel-max-tokens", self.max_tokens),
        ]
        for name, value in pairs:
            if value is None:
                continue
            lines.append(f";; {name}: {value!r}")
        if self.tool_names:
            names = " ".join(f'"{n}"' for n in self.tool_names)
            lines.append(f";; gptel--tool-names: ({names})")
        lines.append(";; End:")
        return "\n".join(lines)

    def save(self, conversation_text: str) -> str | None:
        path = self.session_file()
        if path is None:
            return None
        session_dir().mkdir(parents=True, exist_ok=True)
        content = conversation_text.rstrip("\n") + "\n\n" + self.metadata_block() + "\n"
        tmp = path + ".tmp"
        Path(tmp).write_text(content, encoding="utf-8")
        os.replace(tmp, path)
        return path

    def apply_title(self, title: str) -> None:
        """Rename the session file to <title>_<TS>.md (never overwriting)."""
        title = sanitize_title(title)
        if not title:
            return
        if self.file_path and os.path.exists(self.file_path):
            stamp = time.strftime("%y%m%d%H%M%S")
            new_path = self._unique_path(f"{title}_{stamp}")
            try:
                os.replace(self.file_path, new_path)
                self.file_path = str(new_path)
            except OSError:
                return
        self.title = title

    # -- restoring ---------------------------------------------------------------
    @staticmethod
    def parse_metadata(text: str) -> dict[str, str]:
        """Parse the trailing ;; Local Variables: block (search from EOF).

        Values are stored repr()-style (like the elisp %S printing);
        parsing evaluates them with ast.literal_eval when possible.
        """
        marker = "\n;; Local Variables:\n"
        idx = text.rfind(marker)
        if idx == -1:
            return {}
        block = text[idx + len(marker):]
        end = block.find(";; End:")
        if end != -1:
            block = block[:end]
        meta: dict[str, str] = {}
        for line in block.splitlines():
            line = line.strip()
            if not line.startswith(";; "):
                continue
            body = line[3:]
            if ":" not in body:
                continue
            name, _, value = body.partition(":")
            meta[name.strip()] = _parse_metadata_value(value.strip())
        return meta

    @staticmethod
    def strip_metadata(text: str) -> str:
        marker = "\n;; Local Variables:\n"
        idx = text.rfind(marker)
        if idx == -1:
            return text
        end = text.find(";; End:", idx)
        if end != -1:
            end += len(";; End:")
            return text[:idx] + text[end:]
        return text[:idx]

    @staticmethod
    def latest_session() -> str | None:
        d = session_dir()
        if not d.is_dir():
            return None
        files = sorted(d.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        return str(files[0]) if files else None

    @staticmethod
    def list_sessions() -> list[str]:
        d = session_dir()
        if not d.is_dir():
            return []
        return sorted(
            (str(p) for p in d.glob("*.md")),
            key=lambda p: os.path.getmtime(p),
            reverse=True,
        )


def _parse_metadata_value(value: str) -> str:
    """Parse a repr()-style metadata value, mirroring elisp read-from-string."""
    import ast

    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, str):
            return parsed
        if isinstance(parsed, (list, tuple)):
            return " ".join(str(x) for x in parsed)
        return str(parsed)
    except (ValueError, SyntaxError):
        return value.strip("\"'")
