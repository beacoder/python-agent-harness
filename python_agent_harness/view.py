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
