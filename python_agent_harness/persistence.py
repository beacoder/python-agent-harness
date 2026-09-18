"""Session persistence: auto-save, LLM titles, restore.

Ported from gptel-agent-harness-session.el.

- Auto-save the conversation after each LLM response to
  ~/.local/share/python-agent-harness/sessions/<name>_<YYMMDDHHMMSS>.md
  with a trailing metadata block (;; Local Variables: ...).
- Async title generation from the first user message (title.md);
  on success the file is renamed to <title>_<TS>.md.
- restore / restore-latest commands re-load a session.
"""

from __future__ import annotations

import ast
import os
import re
import threading
import time
from pathlib import Path

from . import config


def session_dir() -> Path:
    return config.SESSION_DIR / config.SESSION_SUBDIR


# The roles the save format delimits blocks with (`**<role>**: `).
SAVED_ROLES = ("user", "assistant", "system", "tool")


def split_role_header(line: str) -> tuple[str, str] | None:
    """``(role, rest)`` when LINE is a ``**role**: `` block header, else None.

    The single source of truth for the save format's block delimiter:
    the renderer escapes what this would match and the parser splits on
    exactly what this accepts, so the two can never drift apart.
    """
    if not line.startswith("**") or "**: " not in line:
        return None
    prefix, _, rest = line.partition("**: ")
    role = prefix.strip("*").strip()
    return (role, rest) if role in SAVED_ROLES else None


def _is_escapable(line: str) -> bool:
    """Whether LINE is a block header, or an already-escaped one."""
    return split_role_header(line.lstrip("\\")) is not None


def escape_role_headers(body: str) -> str:
    r"""Backslash-escape message-body lines that look like block headers.

    Blocks are delimited by ``**<role>**: `` at the start of a line, so a
    message whose own text contains such a line — the agent explaining
    this very format, or a pasted transcript — would otherwise be split
    into extra (and misattributed) messages on restore.  ``\**user**: ``
    still reads as the literal text in markdown and is reversed by
    `unescape_role_header`; already-escaped lines gain another backslash
    so the round trip is exact at any nesting depth.
    """
    if "**" not in body:
        return body
    return "\n".join("\\" + ln if _is_escapable(ln) else ln for ln in body.split("\n"))


def unescape_role_header(line: str) -> str:
    r"""Reverse one level of `escape_role_headers` for a single line.

    Only lines that would otherwise be read as block headers are
    touched, so a literal ``\**note**: `` in a message survives intact.
    """
    return line[1:] if line.startswith("\\") and _is_escapable(line) else line


def sanitize_title(title: str) -> str:
    """Sanitize a generated title (mirrors the elisp semantics)."""
    t = title.strip()
    t = re.sub(r"[\n\r]+", " ", t)
    t = t.strip('"')
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


class SessionPersistence:
    """Saves/restores sessions for one agent session."""

    def __init__(
        self,
        project_dir: str,
        model: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tool_names: list[str] | None = None,
        round_times: list[float] | None = None,
        agent: str | None = None,
    ) -> None:
        self.project_dir = project_dir
        self.model = model
        self.agent = agent
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tool_names = tool_names or []
        self.title: str | None = None
        self.file_path: str | None = None
        self.title_pending = False
        self._first_user_msg: str | None = None
        # wall-clock start times of each round, persisted in the
        # metadata block so restored sessions keep their round
        # timestamps (populated by the TUI on each run)
        self.round_times: list[float] = list(round_times) if round_times else []
        # serializes save vs. apply_title: the rename must never
        # interleave with a save's write+replace, or the conversation
        # would split across two files (a titled stale file plus a
        # fresh untitled one) when title generation races a new run's
        # auto-save
        self._io_lock = threading.Lock()

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
            ("python-agent-harness--model", self.model),
            ("python-agent-harness--agent", self.agent),
        ]
        for name, value in pairs:
            if value is None:
                continue
            lines.append(f";; {name}: {value!r}")
        if self.round_times:
            stamps = " ".join(repr(float(t)) for t in self.round_times)
            lines.append(f";; python-agent-harness--round-times: {stamps}")
        lines.append(";; End:")
        return "\n".join(lines)

    def save(self, conversation_text: str) -> str | None:
        # under the IO lock so a concurrent apply_title rename cannot
        # land between the tmp write and the os.replace (see _io_lock)
        with self._io_lock:
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
        # under the IO lock: the rename must not interleave with a
        # concurrent save's write+replace (see _io_lock)
        with self._io_lock:
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
        block = text[idx + len(marker) :]
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
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, str):
            return parsed
        if isinstance(parsed, (list, tuple)):
            return " ".join(str(x) for x in parsed)
        return str(parsed)
    except (ValueError, SyntaxError):
        return value.strip("\"'")


def parse_saved_body(body: str) -> list:
    """Parse a saved session body back into Message objects.

    The save format is markdown with **role**: content blocks
    separated by blank lines.

    ``tool`` and ``system`` blocks are dropped: the saved markdown
    does not keep ``tool_call_id``/``name`` (assistant tool calls are
    flattened to plain text), so a restored ``role="tool"`` message
    would form an API-invalid payload (a tool message with no
    preceding assistant ``tool_calls``).  A restored ``system``
    message would duplicate the live system prompt the client
    prepends on every request.  The following assistant reply
    already summarizes the results, so dropping them loses no
    essential context.

    Body lines that merely look like a block header are escaped by
    the renderer (see `escape_role_headers`) and unescaped here, so
    a message quoting this format no longer splits into extra
    messages.  Sessions saved before escaping existed can still
    split — that ambiguity is in the file, not in this parser.
    """
    from .attachments import reattach_images
    from .models import Message, TextPart

    messages: list = []
    current_role: str | None = None
    current_lines: list[str] = []

    def _flush(role: str, lines: list[str]) -> None:
        content = "\n".join(lines).strip()
        if not content:
            return
        # A leading image-attachment placeholder (written on save)
        # is re-attached when the file still exists, so the restored
        # message is multimodal again; otherwise it stays as text.
        new_content, parts = reattach_images(content)
        if parts:
            remainder = new_content.split("\n", 1)
            rest_text = remainder[1].strip() if len(remainder) > 1 else ""
            content_parts: list = []
            if rest_text:
                content_parts.append(TextPart(text=rest_text))
            content_parts.extend(parts)
            messages.append(Message(role=role, content=content_parts))
        else:
            messages.append(Message(role=role, content=content))

    for line in body.splitlines():
        # Check for a role header: **user**: ... or **assistant**: ...
        header = split_role_header(line)
        if header is not None:
            role, rest = header
            # Save the previous block (tool blocks lose their
            # tool_call_id/name; system blocks would duplicate the
            # live system prompt the client prepends per request)
            if current_role is not None and current_role not in ("tool", "system"):
                _flush(current_role, current_lines)
            current_role = role
            current_lines = [unescape_role_header(rest)]
            continue
        current_lines.append(unescape_role_header(line))

    # Don't forget the last block (tool/system blocks dropped, see above)
    if current_role is not None and current_role not in ("tool", "system"):
        _flush(current_role, current_lines)

    return messages


def find_session_by_title(query: str) -> str | None:
    """Find a session file by title substring (case-insensitive).

    Matches against the full filename, the filename without .md,
    and the derived title.  Returns the most recent match, or None.
    """
    query_lower = query.lower()
    # Strip .md from query if present, for cleaner substring matching
    query_stem = query_lower[:-3] if query_lower.endswith(".md") else query_lower
    files = SessionPersistence.list_sessions()  # already sorted by mtime desc
    for f in files:
        basename = os.path.basename(f)
        basename_lower = basename.lower()
        # Exact basename match (with or without .md)
        if basename_lower == query_lower or basename_lower == query_lower + ".md":
            return f
        # Substring match against filename (minus .md)
        name_part = basename[:-3] if basename.endswith(".md") else basename
        if query_stem in name_part.lower():
            return f
        # Match against derived title (dashes → spaces)
        title = title_from_filename(f)
        if title and query_stem in title.lower():
            return f
    return None
