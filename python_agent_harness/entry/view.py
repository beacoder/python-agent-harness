"""View protocol for the agent harness.

Defines the contract a presentation layer must satisfy to drive the
agent.  The controller (``Controller``) talks to the model (``Session``)
and pushes events / requests to whatever ``View`` is attached, so the
concrete presentation (the Rich TUI today, a headless/CI view tomorrow)
is swappable without touching the model or the controller.

The protocol is structural (``typing.Protocol``): a class satisfies it by
implementing the methods, no inheritance required.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..io.text_filter import final_check_hold_index, strip_final_check


class FilteredDeltaStream:
    """Turns streamed assistant text into append-only chunks that carry
    the same content the TUI displays (TUI parity).

    The model ends a reply with a [FINAL CHECK] block (Goal/Status/
    Evidence) as verification bookkeeping, and the TUI filters it out.
    The TUI can re-render its whole buffer every frame, so it may show a
    half-streamed block and then drop it; a delta stream cannot — a
    chunk that has been written is on the wire for good.

    So this holds back the tail that a later chunk might delete: a
    block-in-progress and the whitespace in front of it (see
    ``final_check_hold_index``).  The held text is released as soon as
    the next chunk rules the block out, or by ``flush`` at the end of a
    message.  Concatenating everything ``feed``/``flush`` return
    therefore equals ``strip_final_check`` over the whole message.

    ``flush`` ends a message (a committed reply before a tool call, a
    discarded partial before a retry, or the end of a run) and returns
    the held remainder; ``reset`` drops the accumulation without
    emitting.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._emitted = 0

    def feed(self, text: str) -> str:
        """Append *text*; return the newly emittable chunk (may be empty)."""
        self._buf += text
        # The emitted prefix is proven block-free (a block starting there
        # would have been held), so both scans may skip it -- that is what
        # keeps a long reply from costing O(n^2) as chunks arrive.
        visible = strip_final_check(self._buf, self._emitted)
        safe = final_check_hold_index(visible, self._emitted)
        if safe > self._emitted:
            chunk = visible[self._emitted : safe]
            self._emitted = safe
            return chunk
        return ""

    def flush(self) -> str:
        """End the message: return whatever was held back, then reset.

        A header that never completed into a block is real content, so
        it must still reach the stream.
        """
        visible = strip_final_check(self._buf, self._emitted)
        chunk = visible[self._emitted :] if len(visible) > self._emitted else ""
        self.reset()
        return chunk

    def reset(self) -> None:
        self._buf = ""
        self._emitted = 0


@runtime_checkable
class View(Protocol):
    """A presentation layer that renders agent activity and collects input.

    The controller wires the session's callbacks to these methods (see
    ``Controller.attach_view``) and calls ``run`` to start the main
    loop.  All methods may be invoked from the agent's worker thread
    except ``run`` (main thread) and the interactive ``confirm``/``ask``
    (which block until the user answers).
    """

    def on_delta(self, text: str) -> None:
        """A chunk of streamed assistant text arrived."""
        ...

    def on_notify(self, kind: str, data: Any = None) -> None:
        """A structured event (tool_start, tools, compact, error, ...)."""
        ...

    def on_log(self, msg: str) -> None:
        """A free-form status/log line."""
        ...

    def confirm(self, prompt: str) -> bool:
        """Ask the user a yes/no question; return their decision."""
        ...

    def ask(self, questions: list[dict]) -> str:
        """Ask the user one or more questions; return the combined answer."""
        ...

    def run(self) -> None:
        """Start the view's main loop (blocking)."""
        ...
