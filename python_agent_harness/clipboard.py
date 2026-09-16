"""Clipboard image capture: read an image off the OS clipboard and
write it to a temporary file so it can flow through the existing
``@path`` attachment path.

A terminal paste delivers *text*, never raw image bytes — the OS
clipboard image never enters the terminal input stream.  So when the
user pastes, the TUI inspects the OS clipboard out-of-band: if it holds
an image, ``grab_clipboard_image`` shells out to the platform's
clipboard tool (no new runtime dependency, preserving the project's
"three runtime dependencies" philosophy), writes the bytes to a temp
file under the shared spool dir, and returns the path.  The caller
turns that into an ``@<path>`` token; validation, ``ImagePart``
creation, and temp-file cleanup all reuse the existing machinery.

Supported tools (first available wins per platform):
    - macOS:   ``pngpaste`` (writes PNG to a file)
    - Wayland: ``wl-paste`` (``--type image/png``)
    - X11:     ``xclip`` (``-selection clipboard -t image/png -o``)
    - Windows: PowerShell ``Get-Clipboard -Format Image``

Every path is best-effort: any failure (no tool installed, no image on
the clipboard, empty/garbage output) returns ``None`` so the caller
falls back to normal text paste.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
import tempfile
import time

# The magic-byte prefix ``attachments`` validates PNG against; captured
# clipboard images are always written as PNG, so a quick sanity check
# here rejects empty / non-image tool output before touching disk.
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# Cap on captured image size (bytes).  Mirrors ``attachments.MAX_IMAGE_SIZE``
# (20 MB) so a capture that would fail validation is rejected up front.
_MAX_CAPTURE = 20 * 1024 * 1024

# How long to wait for a clipboard tool before giving up (seconds).
# Kept short: the capture runs synchronously on the TUI key-binding
# thread, so a long timeout would freeze the UI on an empty paste.
_TOOL_TIMEOUT = 2.0


def _spool_dir() -> str:
    """Temp dir for the captured image; reuse the tools' spool dir so
    cleanup and TMPDIR handling stay consistent."""
    from .tools.filesystem import _spool_dir as fs_spool_dir

    return fs_spool_dir()


def _track_temp_file(path: str) -> None:
    """Register PATH for best-effort deletion on session close, reusing
    the existing spooled-file cleanup list (``cleanup_spooled_files``)."""
    from .tools import filesystem

    with filesystem._spooled_files_lock:
        filesystem._spooled_files.append(path)


def _run(cmd: list[str]) -> bytes | None:
    """Run CMD, returning stdout bytes on success, ``None`` on any error.

    A non-zero exit (no image on the clipboard) or a missing binary both
    yield ``None`` — the caller treats every failure as "no clipboard
    image" and falls back to text paste.
    """
    if shutil.which(cmd[0]) is None:
        return None
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=_TOOL_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not proc.stdout:
        return None
    return proc.stdout


def _capture_png_bytes() -> bytes | None:
    """Read an image off the clipboard as PNG bytes, or ``None``.

    Tries the platform-appropriate tool(s) in order; the first that
    yields non-empty PNG-signed output wins.
    """
    if sys.platform == "darwin":
        # pngpaste writes the clipboard image (as PNG) to stdout with "-".
        data = _run(["pngpaste", "-"])
        return data
    if sys.platform == "win32":
        return _capture_windows()
    # Linux / *nix: choose ONE tool by environment to avoid chaining
    # several blocking subprocesses on the UI thread.  A Wayland session
    # (WAYLAND_DISPLAY) uses wl-paste; otherwise xclip.  Fall back to the
    # other tool only when the chosen one isn't installed at all (not on
    # an empty result), so a missing image never fans out into multiple
    # slow spawns.
    on_wayland = bool(os.environ.get("WAYLAND_DISPLAY"))
    primary = (
        ["wl-paste", "--type", "image/png"]
        if on_wayland
        else ["xclip", "-selection", "clipboard", "-t", "image/png", "-o"]
    )
    secondary = (
        ["xclip", "-selection", "clipboard", "-t", "image/png", "-o"]
        if on_wayland
        else ["wl-paste", "--type", "image/png"]
    )
    if shutil.which(primary[0]) is not None:
        return _run(primary)
    return _run(secondary)


def _capture_windows() -> bytes | None:
    """Capture a clipboard image on Windows via PowerShell.

    PowerShell saves the clipboard image to a temp PNG (System.Windows.
    Forms.Clipboard), which we then read back.  Returns ``None`` when no
    image is on the clipboard or PowerShell is unavailable.
    """
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if powershell is None:
        return None
    fd, tmp = tempfile.mkstemp(suffix=".png", dir=_spool_dir())
    os.close(fd)
    # -STA is required for clipboard access; the script exits non-zero
    # when the clipboard holds no image so _run-style handling applies.
    # The path goes into a single-quoted PowerShell literal, so any
    # single quote in the temp dir (e.g. C:\Users\O'Brien\Temp) must be
    # doubled ('' is an escaped ' inside a PS single-quoted string) or
    # the script would fail to parse.
    ps_path = tmp.replace("'", "''")
    script = (
        "Add-Type -AssemblyName System.Windows.Forms;"
        "$img = [System.Windows.Forms.Clipboard]::GetImage();"
        "if ($img -eq $null) { exit 1 };"
        f"$img.Save('{ps_path}', [System.Drawing.Imaging.ImageFormat]::Png)"
    )
    try:
        proc = subprocess.run(
            [powershell, "-NoProfile", "-STA", "-Command", script],
            capture_output=True,
            timeout=_TOOL_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        _silent_remove(tmp)
        return None
    if proc.returncode != 0:
        _silent_remove(tmp)
        return None
    try:
        with open(tmp, "rb") as f:
            data = f.read()
    except OSError:
        data = None
    _silent_remove(tmp)
    return data or None


def _silent_remove(path: str) -> None:
    with contextlib.suppress(OSError):
        os.remove(path)


def grab_clipboard_image() -> str | None:
    """Capture a clipboard image to a temp PNG file; return its path.

    Returns ``None`` when the clipboard holds no image, no clipboard
    tool is available, or the captured bytes fail a basic PNG sanity
    check / size cap.  The temp file is tracked for cleanup on session
    close (shared with the tool-output spool list).
    """
    data = _capture_png_bytes()
    if not data:
        return None
    if len(data) > _MAX_CAPTURE:
        return None
    if not data.startswith(_PNG_MAGIC):
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    try:
        fd, path = tempfile.mkstemp(
            prefix=f"python-agent-harness-clip-{stamp}-",
            suffix=".png",
            dir=_spool_dir(),
        )
        with os.fdopen(fd, "wb") as f:
            f.write(data)
    except OSError:
        return None
    _track_temp_file(path)
    return path
