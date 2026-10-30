"""Terminal mouse support for the live conversation panel.

Clicking a ``▶ tool: ...`` row expands it into its full multi-line form
(and clicking again collapses it).  A terminal can only report clicks
for a region the application is actively drawing, so mouse reporting is
enabled for exactly as long as the panel is live (during a run) and is
always turned back off afterwards — including on error — so the user's
terminal never ends up printing escape sequences for every click.

Protocol: xterm mode 1000 (press/release reporting) plus 1006 (SGR
coordinates, which report columns/rows beyond 223 unambiguously).
Reports arrive on stdin as ``ESC [ < b ; col ; row M|m``.  Reading them
means putting the terminal in cbreak mode, which also hands us any
keystroke typed while the agent runs; those are collected as plain text
and replayed into the next input prompt (see ``pending_text``) instead
of being silently dropped.

A report only says *which row of the terminal* was clicked, so the
panel's own position on screen is asked for with a Device Status Report
(``ESC [ 6 n``) at click time — see ``cursor_row``.
"""

from __future__ import annotations

import contextlib
import os
import re
import select
import termios
import threading
import tty
from typing import Any

ENABLE = "\x1b[?1000h\x1b[?1006h"
DISABLE = "\x1b[?1006l\x1b[?1000l"

# Device Status Report (DSR): the terminal answers with ESC [ row ; col R.
# Used to locate the live panel on screen: a click reports a terminal
# row, and the panel is not always bottom-anchored (rich's Live repaints
# from wherever the cursor was), so the frame's top row has to be asked
# for rather than assumed.
CURSOR_QUERY = "\x1b[6n"

# SGR (1006) mouse report: ESC [ < button ; column ; row (M press, m release)
_SGR = re.compile(rb"\x1b\[<(\d+);(\d+);(\d+)([Mm])")

# DSR reply: ESC [ row ; col R  (also stripped when unsolicited, so a
# late answer can never reach the input prompt as typed text)
_DSR = re.compile(rb"\x1b\[(\d+);(\d+)R")

# A partially received report/escape sequence at the end of a read; kept
# for the next chunk instead of being decoded as text.
_MAX_CARRY = 32

LEFT_BUTTON = 0


def parse_events(data: bytes) -> list[tuple[int, int, int, str]]:
    """Complete mouse reports in DATA as ``(button, column, row, kind)``.

    ``kind`` is ``"press"`` for the terminal's ``M`` suffix and
    ``"release"`` for ``m``.  Coordinates are 1-based, exactly as the
    terminal sent them.
    """
    events = []
    for m in _SGR.finditer(data):
        events.append(
            (
                int(m.group(1)),
                int(m.group(2)),
                int(m.group(3)),
                "press" if m.group(4) == b"M" else "release",
            )
        )
    return events


def is_left_click(button: int) -> bool:
    """True for a plain left-button press.

    Wheel (64+), drag/release-with-motion (32+ button mask) and
    modifier bits (4/8/16) are excluded: only a bare left click is a
    toggle request.
    """
    return button == LEFT_BUTTON


def take_cursor_reports(data: bytes) -> tuple[bytes, list[tuple[int, int]]]:
    """Split DATA into (data without DSR replies, [(row, column), ...]).

    Replies are removed unconditionally: an answer that arrives after
    the click handler stopped waiting would otherwise be decoded as
    typed text and land in the next input prompt.
    """
    return _DSR.sub(b"", data), [(int(m.group(1)), int(m.group(2))) for m in _DSR.finditer(data)]


def split(data: bytes) -> tuple[list[tuple[int, int, int, str]], bytes, str]:
    """Split a raw stdin chunk into (events, carry, text).

    ``carry`` is an incomplete trailing escape sequence to prepend to
    the next chunk; ``text`` is whatever the user typed while the
    mouse reader owned the terminal.
    """
    events = parse_events(data)
    rest = _SGR.sub(b"", data)
    carry = b""
    esc = rest.find(b"\x1b")
    if esc != -1:
        tail = rest[esc:]
        if len(tail) <= _MAX_CARRY:
            carry = tail
            rest = rest[:esc]
    text = rest.decode("utf-8", "ignore")
    # drop control characters (arrow keys, Ctrl-anything): the input
    # prompt owns those keys, we only preserve ordinary typing
    text = "".join(ch for ch in text if ch.isprintable())
    return events, carry, text


class MouseSession:
    """Enable terminal mouse reporting + deliver clicks for a live panel.

    ``on_click(column, row)`` is called (on the reader thread) for every
    left-button press, with 1-based terminal cell coordinates.
    ``on_text(text)`` receives characters typed while reporting is on.
    """

    def __init__(
        self,
        console: Any,
        on_click: Any,
        on_text: Any = None,
    ) -> None:
        self.console = console
        self.on_click = on_click
        self.on_text = on_text
        self.pending_text = ""
        self._fd: int = 0
        self._saved_attrs: list[Any] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._active = False
        self._lock = threading.Lock()
        # last cursor row reported by the terminal (DSR reply)
        self._cursor_row: int | None = None
        self._cursor_seen = threading.Event()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def supported(self) -> bool:
        """Whether this process can read mouse reports at all.

        Dumb terminals (``TERM=dumb``/unset) and non-tty stdin (piped
        input, the test suite) have no terminal to put in mouse mode.
        """
        if getattr(self.console, "is_dumb_terminal", False):
            return False
        try:
            fd = self.console.file.fileno()
        except (AttributeError, OSError, ValueError):
            return False
        if not os.isatty(fd):
            return False
        try:
            if not os.isatty(0):
                return False
        except OSError:
            return False
        return True

    def enable(self) -> bool:
        """Turn mouse reporting on and start the reader thread."""
        with self._lock:
            if self._active or not self.supported():
                return self._active
            try:
                self._fd = 0
                self._saved_attrs = termios.tcgetattr(0)
                # TCSANOW: never block on output, and never discard
                # typeahead the user already typed (it is captured and
                # replayed into the next prompt instead)
                tty.setcbreak(0, termios.TCSANOW)
                self._write(ENABLE)
            except Exception:
                # No mouse support on this terminal: the panel still
                # renders, clicks are simply never delivered.
                self._restore_terminal()
                return False
            self._stop.clear()
            self._active = True
            self._thread = threading.Thread(target=self._read_loop, name="tui-mouse", daemon=True)
            self._thread.start()
            return True

    def disable(self) -> None:
        """Stop reading and restore the terminal's original modes."""
        with self._lock:
            if not self._active:
                self._restore_terminal()
                return
            self._active = False
            self._stop.set()
            thread = self._thread
            self._thread = None
        if thread is not None and thread.is_alive():
            # the reader wakes from select within its timeout, so this
            # returns promptly; joining guarantees it can never steal a
            # keystroke from the input prompt that starts right after
            thread.join(timeout=1.0)
        with self._lock:
            self._restore_terminal()

    def __enter__(self) -> MouseSession:
        self.enable()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.disable()

    def cursor_row(self, timeout: float = 0.2) -> int | None:
        """Ask the terminal which row the cursor is on (1-based).

        The answer is needed to place the live panel on screen: a click
        reports an absolute terminal row, and rich's ``Live`` repaints
        from wherever the cursor happens to be, so the frame is only
        bottom-anchored when it actually fills the screen.  The query is
        written under the console lock so it cannot land in the middle
        of a repaint (where the cursor sits on the frame's first row
        instead of below its last one).

        Returns None when reporting is off or the terminal does not
        answer (the caller then falls back to assuming a full panel).
        """
        if not self._active:
            return None
        self._cursor_row = None
        self._cursor_seen.clear()
        self._write(CURSOR_QUERY)
        if not self._cursor_seen.wait(timeout):
            return None
        return self._cursor_row

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _restore_terminal(self) -> None:
        if self._saved_attrs is not None:
            # TCSANOW, not TCSADRAIN: draining waits for the panel's
            # pending output to be consumed by the terminal, which can
            # block forever if the terminal is busy — a run must never
            # hang on its way out.
            with contextlib.suppress(Exception):
                termios.tcsetattr(0, termios.TCSANOW, self._saved_attrs)
            self._saved_attrs = None
        self._write(DISABLE)

    def _write(self, seq: str) -> None:
        # under the console lock: rich repaints the panel from another
        # thread, and a control sequence spliced into the middle of a
        # repaint would be drawn as garbage on screen
        lock = getattr(self.console, "_lock", None)
        with contextlib.suppress(AttributeError, OSError, ValueError):
            if lock is not None:
                lock.acquire()
            try:
                self.console.file.write(seq)
                self.console.file.flush()
            finally:
                if lock is not None:
                    lock.release()

    def _read_loop(self) -> None:
        carry = b""
        while not self._stop.is_set():
            try:
                ready, _, _ = select.select([self._fd], [], [], 0.05)
            except (OSError, ValueError):
                return
            if not ready:
                continue
            try:
                data = os.read(self._fd, 1024)
            except OSError:
                return
            if not data:
                return
            data, cursors = take_cursor_reports(carry + data)
            if cursors:
                self._cursor_row = cursors[-1][0]
                self._cursor_seen.set()
            events, carry, text = split(data)
            if text:
                self.pending_text += text
                if self.on_text is not None:
                    self.on_text(text)
            for button, col, row, kind in events:
                if kind == "press" and is_left_click(button):
                    # off this thread: the handler asks the terminal for
                    # the cursor row, which only this thread can read
                    threading.Thread(
                        target=self._fire_click,
                        args=(col, row),
                        name="tui-mouse-click",
                        daemon=True,
                    ).start()

    def _fire_click(self, column: int, row: int) -> None:
        """Deliver one click, never letting a handler break the run."""
        with contextlib.suppress(Exception):
            self.on_click(column, row)
