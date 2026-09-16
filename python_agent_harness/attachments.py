"""File attachment utilities: @file parsing, validation, and part creation.

The TUI detects ``@path`` references in user input, validates the file
(e.g. image type, size), and converts it into the appropriate content
part (``ImagePart`` for images, ``TextPart`` for text files).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from .models import ImagePart, TextPart

# Maximum image file size: 20 MB (generous; most APIs accept up to ~20 MB).
MAX_IMAGE_SIZE = 20 * 1024 * 1024

# Maximum text file size: 512 KB (larger files would blow up the context
# window; the model can use the Read tool for targeted access instead).
MAX_TEXT_SIZE = 512 * 1024

# Supported image MIME types by extension.
IMAGE_EXTENSIONS: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

# Magic-byte signatures for the supported image types, used to reject
# files whose extension lies about their content (a renamed .txt or a
# truncated download).  Each entry is (offset, bytes) — the signature
# must match at that offset.  JPEG has no single fixed signature: it
# starts with an SOI marker (FF D8) and ends with EOI (FF D9).
_IMAGE_SIGNATURES: dict[str, tuple[tuple[int, bytes], ...]] = {
    "image/png": ((0, b"\x89PNG\r\n\x1a\n"),),
    "image/jpeg": ((0, b"\xff\xd8"),),
    "image/gif": ((0, b"GIF87a"), (0, b"GIF89a")),
    "image/webp": ((0, b"RIFF"), (8, b"WEBP")),
}

# Text-like file extensions (read as UTF-8 text).
TEXT_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".java",
        ".c",
        ".cpp",
        ".cc",
        ".h",
        ".hpp",
        ".rs",
        ".go",
        ".rb",
        ".php",
        ".swift",
        ".kt",
        ".scala",
        ".sh",
        ".bash",
        ".zsh",
        ".fish",
        ".ps1",
        ".md",
        ".rst",
        ".txt",
        ".log",
        ".csv",
        ".tsv",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".conf",
        ".xml",
        ".html",
        ".css",
        ".scss",
        ".less",
        ".sql",
        ".graphql",
        ".proto",
        ".env",
        ".gitignore",
        ".dockerignore",
        ".makefile",
        ".cmake",
    }
)

# Files without an extension that should be treated as text.
TEXT_BASENAMES: frozenset[str] = frozenset(
    {
        "Makefile",
        "Dockerfile",
        "Rakefile",
        "Gemfile",
        "LICENSE",
        "README",
        "CHANGELOG",
        ".gitignore",
        ".dockerignore",
        ".env",
    }
)


# Placeholder written into saved sessions for image attachments.  The
# image bytes themselves are not persisted; the source path(s) are
# recorded so a restored session can re-attach the image when the file
# still exists.
IMAGE_PLACEHOLDER_RE = re.compile(
    r"^\[(\d+) image attachment\(s\)(?: from ([^\n]+))?"
    r" — (not available in restored session|re-attached on restore)\]"
    r"(?=\n|$)"
)


def _format_paths(paths: list[str]) -> str:
    """Join paths with ', ', escaping backslashes and commas so the
    list can be split back exactly (paths may contain commas)."""
    return ", ".join(p.replace("\\", "\\\\").replace(",", "\\,") for p in paths)


def _split_paths(raw: str) -> list[str]:
    """Reverse of ``_format_paths``: split on unescaped commas."""
    out: list[str] = []
    cur = ""
    i = 0
    while i < len(raw):
        c = raw[i]
        if c == "\\" and i + 1 < len(raw) and raw[i + 1] in ",\\":
            cur += raw[i + 1]
            i += 2
        elif c == ",":
            out.append(cur.strip())
            cur = ""
            i += 1
        else:
            cur += c
            i += 1
    out.append(cur.strip())
    return [p for p in out if p]


def image_placeholder(count: int, paths: list[str] | None = None) -> str:
    """Build the image-attachment placeholder line for session saves."""
    loc = f" from {_format_paths(paths)}" if paths else ""
    return f"[{count} image attachment(s){loc} — not available in restored session]"


# Visual marker inserted into the input buffer when a clipboard image is
# captured on paste.  The image itself is tracked out-of-band (not via an
# @path token), so this marker is purely cosmetic and is stripped from
# the text before the message is built.  Kept here so the producer (TUI
# paste handler) and consumer (submit path) agree on one format.
def clipboard_image_marker(path: str) -> str:
    """The buffer marker for a pasted clipboard image at PATH."""
    return f"[image #{os.path.basename(path)}] "


def strip_clipboard_markers(text: str, paths: list[str]) -> str:
    """Remove the clipboard-image markers for PATHS from TEXT.

    Only the exact marker strings for the given pending paths are
    removed (one occurrence per path), so text a user literally typed
    that merely resembles a marker (e.g. ``data[image #2]`` in prose) is
    never touched.  Markers are stripped in path order; a trailing space
    left dangling by the removal is not collapsed (the surrounding text
    is otherwise preserved verbatim).
    """
    for path in paths:
        marker = clipboard_image_marker(path)
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx] + text[idx + len(marker) :]
    return text


def load_clipboard_image(path: str) -> ParsedAttachment | AttachmentError:
    """Validate a captured clipboard image PATH into a ``ParsedAttachment``.

    Reuses the same validation as ``@file`` images (existence, size cap,
    image signature) so a pasted image is held to the same standard.
    """
    return _validate_and_create(path, os.path.basename(path))


@dataclass
class AttachmentError:
    """A validation error for a file attachment."""

    path: str
    message: str


@dataclass
class ParsedAttachment:
    """A successfully parsed file attachment."""

    path: str
    part: TextPart | ImagePart


def parse_at_references(
    text: str, project_dir: str
) -> tuple[str, list[ParsedAttachment], list[AttachmentError]]:
    """Find ``@path`` references in TEXT and resolve them to content parts.

    Returns ``(cleaned_text, attachments, errors)``:

    - ``cleaned_text``: the input text with ``@path`` tokens replaced by
      just the path (the ``@`` stripped), so the model sees the file
      content in the attachment, not a duplicate in the text.
    - ``attachments``: successfully parsed ``ParsedAttachment`` objects
      (one per valid ``@path``), in order of appearance.
    - ``errors``: validation errors for files that could not be attached.

    Path resolution: relative paths are resolved against ``project_dir``;
    ``~`` is expanded; absolute paths are used as-is.

    Only ``@`` followed by a non-whitespace path-like token is treated as
    a reference.  ``@`` at the start of the input or after a space is
    a reference; ``@`` inside an email address or word (e.g. ``user@host``)
    is NOT treated as a reference.
    """
    # Match @path where @ is at start of text, preceded by whitespace,
    # or preceded by an opening bracket ("(@file.png)").  The path token
    # is non-whitespace, non-empty, and must contain at least one
    # path-like character (/, ., ~, or alphanumeric).
    #
    # Trailing punctuation is NOT part of the path: a sentence-final
    # "@file.png." or a comma-separated "@a.png, @b.png" must resolve
    # the bare path.  A closing paren/bracket/quote directly after the
    # path is likewise stripped (e.g. "(@file.png)").  The stripped
    # characters stay in the text — only the @ and the path token are
    # replaced.
    pattern = re.compile(r"(?:^|[\s(\[{])@([^\s@]+)")
    attachments: list[ParsedAttachment] = []
    errors: list[AttachmentError] = []
    replacements: list[tuple[str, str]] = []

    for match in pattern.finditer(text):
        raw_path = match.group(1)
        # Strip trailing sentence/closing punctuation from the path
        # token: "@file.png." / "@file.png," / "(@file.png)" must
        # resolve "file.png", not "file.png." / "file.png," / "file.png)".
        # The stripped characters are left in the text (the replacement
        # below only swaps the @ + path token).
        stripped = raw_path.rstrip(".,;:!?)]}>\"'")
        if not stripped:
            continue
        resolved = _resolve_path(stripped, project_dir)
        result = _validate_and_create(resolved, stripped)
        if isinstance(result, AttachmentError):
            errors.append(result)
            # Keep the original @path in the text on error so the user
            # can see what failed
            continue
        attachments.append(result)
        # Replace the @path token (including the leading space/start)
        # with just the path (no @), so the text reads naturally.  The
        # trailing punctuation stripped above stays in the text.
        full_match = match.group(0)
        replacements.append((full_match, full_match.replace("@" + stripped, stripped, 1)))

    cleaned = text
    for old, new in replacements:
        cleaned = cleaned.replace(old, new, 1)

    return cleaned, attachments, errors


def _has_image_signature(data: bytes, media_type: str) -> bool:
    """True when DATA's magic bytes match MEDIA_TYPE's signature.

    Any of the type's signatures matching at its offset is enough
    (e.g. GIF87a or GIF89a for image/gif).  A file shorter than the
    signature offset+length can never match.
    """
    signatures = _IMAGE_SIGNATURES.get(media_type)
    if not signatures:
        return True  # unknown type: no signature to check, accept
    for offset, sig in signatures:
        if len(data) >= offset + len(sig) and data[offset : offset + len(sig)] == sig:
            return True
    return False


def _resolve_path(raw_path: str, project_dir: str) -> str:
    """Resolve a raw @path token to an absolute filesystem path."""
    expanded = os.path.expanduser(raw_path)
    if os.path.isabs(expanded):
        return expanded
    return os.path.join(project_dir, expanded)


def _validate_and_create(resolved: str, display_path: str) -> ParsedAttachment | AttachmentError:
    """Validate a file and create the appropriate content part.

    Returns ``ParsedAttachment`` on success, ``AttachmentError`` on failure.
    """
    if not os.path.isfile(resolved):
        return AttachmentError(display_path, f"file not found: {display_path}")

    ext = Path(resolved).suffix.lower()
    basename = os.path.basename(resolved)

    # Image files
    if ext in IMAGE_EXTENSIONS:
        size = os.path.getsize(resolved)
        if size > MAX_IMAGE_SIZE:
            return AttachmentError(
                display_path,
                f"image too large ({size // 1024 // 1024} MB); max is {MAX_IMAGE_SIZE // 1024 // 1024} MB",
            )
        if size == 0:
            return AttachmentError(display_path, f"image file is empty: {display_path}")
        media_type = IMAGE_EXTENSIONS[ext]
        with open(resolved, "rb") as f:
            data = f.read()
        if not _has_image_signature(data, media_type):
            return AttachmentError(
                display_path,
                f"file does not look like a {media_type} image (content does not match the .{ext.lstrip('.')} extension)",
            )
        return ParsedAttachment(
            resolved, ImagePart(data=data, media_type=media_type, path=resolved)
        )

    # Text files
    if ext in TEXT_EXTENSIONS or basename in TEXT_BASENAMES or not ext:
        size = os.path.getsize(resolved)
        if size > MAX_TEXT_SIZE:
            return AttachmentError(
                display_path,
                f"file too large ({size // 1024} KB); max is {MAX_TEXT_SIZE // 1024} KB "
                f"for text attachments — use the Read tool instead",
            )
        try:
            with open(resolved, encoding="utf-8", errors="replace") as f:
                content = f.read()
        except OSError as e:
            return AttachmentError(display_path, f"cannot read file: {e}")
        return ParsedAttachment(resolved, TextPart(text=content))

    # Unknown extension: try to read as text
    size = os.path.getsize(resolved)
    if size > MAX_TEXT_SIZE:
        return AttachmentError(
            display_path,
            f"file too large ({size // 1024} KB); max is {MAX_TEXT_SIZE // 1024} KB "
            f"for text attachments — use the Read tool instead",
        )
    try:
        with open(resolved, encoding="utf-8", errors="replace") as f:
            content = f.read()
    except OSError as e:
        return AttachmentError(display_path, f"cannot read file: {e}")
    return ParsedAttachment(resolved, TextPart(text=content))


def reattach_images(text: str) -> tuple[str, list[ImagePart]]:
    """Re-attach images from a saved-session placeholder line.

    TEXT must be a message body whose first line is an image-attachment
    placeholder (see ``image_placeholder``).  Returns ``(new_text,
    parts)``: for each recorded path whose file still exists and passes
    validation, the placeholder line is rewritten as a
    ``re-attached on restore`` variant and the rebuilt ``ImagePart``
    objects are returned.  When no image can be re-attached, the text
    is returned unchanged.
    """
    m = IMAGE_PLACEHOLDER_RE.match(text)
    if not m:
        return text, []
    paths = _split_paths(m.group(2) or "")
    parts: list[ImagePart] = []
    for path in paths:
        result = _validate_and_create(path, path)
        if isinstance(result, ParsedAttachment) and isinstance(result.part, ImagePart):
            parts.append(result.part)
    if not parts:
        return text, []
    new_line = (
        f"[{len(parts)} image attachment(s) from "
        f"{_format_paths([p.path for p in parts if p.path])} — re-attached on restore]"
    )
    return new_line + text[m.end() :], parts
