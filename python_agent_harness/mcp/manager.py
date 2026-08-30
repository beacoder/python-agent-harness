"""MCPManager: lifecycle + one-time tool discovery for MCP servers.

Synchronous facade over the async MCPClient wrapper: the harness's
agent loop is synchronous, so every SDK interaction runs on a DEDICATED
event-loop thread (``asyncio.run`` per call would bind SDK resources —
subprocess pipes, anyio memory streams — to a fresh loop each time and
break on the next call).

Lifecycle (mirrors the session lifecycle; see ``Session``)::

    manager = MCPManager(config)
    failures = manager.connect_all()     # connect every configured server
    specs = manager.discover_tools()     # tools/list ONCE per session
    ... register MCPTool instances from the specs ...
    manager.call_tool("github", "search", {...})
    manager.close_all()

The agent loop never sees this class: MCP tools are ordinary registry
tools, and the manager only ever returns plain dicts or raises the
SDK/connection errors that the tool adapter turns into error strings.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import threading
import time
from collections.abc import Callable, Coroutine
from typing import Any

from .client import MCPClient, MCPUnavailableError
from .config import MCPConfig


class MCPCallCancelled(Exception):
    """Raised when an in-flight MCP call is cancelled (Ctrl-C).

    The manager's cancel check is polled while waiting for the SDK
    call; when it fires the underlying future is cancelled and this
    exception propagates, so a hung server call can never wedge the
    agent-loop thread forever.
    """


class MCPToolSpec:
    """A tool advertised by an MCP server (one tools/list entry)."""

    __slots__ = ("server", "name", "description", "input_schema")

    def __init__(
        self,
        server: str,
        name: str,
        description: str,
        input_schema: dict[str, Any],
    ) -> None:
        self.server = server
        self.name = name
        self.description = description
        self.input_schema = input_schema


class _LoopThread:
    """A background thread running a persistent asyncio event loop.

    All SDK interactions for one MCPManager run on this loop, so
    transport resources stay bound to a single loop for their whole
    lifetime.  ``run`` blocks the caller until the coroutine completes
    (or TIMEOUT elapses) and re-raises its exception here.
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True, name="mcp-loop")
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(
        self,
        coro: Coroutine[Any, Any, Any],
        timeout: float | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> Any:
        """Run CORO on the loop thread; raise its exception here.

        Waits in short slices instead of one blocking
        ``future.result(timeout=...)``: a ``cancel_check`` is polled
        between slices, so Ctrl-C unblocks a hung SDK call (the future
        is cancelled and :class:`MCPCallCancelled` raised) instead of
        wedging the caller forever.  ``timeout`` is a wall-clock
        deadline; on expiry the future is cancelled and TimeoutError
        raised.  A TimeoutError raised BY the coroutine itself (SDK
        read timeout) propagates untouched — ``future.exception`` keeps
        it distinct from the poll timeout.
        """
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        deadline = time.monotonic() + timeout if timeout is not None else None
        while True:
            try:
                exc = future.exception(timeout=0.1)
            # concurrent.futures.TimeoutError is the builtin TimeoutError on
            # 3.11+ but a distinct class on 3.10 — catch the futures one so
            # the poll-timeout branch works on every supported Python.
            except concurrent.futures.TimeoutError:
                # future still running: enforce deadline / cancellation
                if deadline is not None and time.monotonic() >= deadline:
                    future.cancel()
                    raise TimeoutError(f"MCP call timed out after {timeout}s") from None
                if cancel_check is not None and cancel_check():
                    future.cancel()
                    raise MCPCallCancelled("MCP call cancelled") from None
                continue
            if exc is not None:
                raise exc
            return future.result()

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        # close the loop object itself (it is stopped and idle by now),
        # or the GC reports ResourceWarning for every manager lifetime
        with contextlib.suppress(Exception):  # best effort teardown
            self._loop.close()


class MCPManager:
    """Owns the MCP server connections for one session."""

    def __init__(self, config: MCPConfig | None = None) -> None:
        self.config = config if config is not None else MCPConfig()
        self._loop: _LoopThread | None = None
        self._clients: dict[str, MCPClient] = {}
        self._tools: dict[str, MCPToolSpec] = {}  # "server__tool" -> spec
        self._discovered: set[str] = set()
        # (server, error) pairs from the last connect_all / discovery
        self.errors: list[tuple[str, str]] = []

    # ------------------------------------------------------------------
    # plumbing
    # ------------------------------------------------------------------
    def _ensure_loop(self) -> _LoopThread:
        if self._loop is None:
            self._loop = _LoopThread()
        return self._loop

    def _call(
        self,
        coro: Coroutine[Any, Any, Any],
        timeout: float | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> Any:
        return self._ensure_loop().run(coro, timeout=timeout, cancel_check=cancel_check)

    @property
    def connected(self) -> list[str]:
        """Names of the currently-connected servers (sorted)."""
        return sorted(self._clients)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def connect_all(self) -> list[tuple[str, str]]:
        """Connect every configured server; return ``[(name, error)]`` failures.

        A failing server never takes the session down: it is skipped
        (with its error recorded) and the rest keep working.  Idempotent:
        already-connected servers are left alone.
        """
        failures: list[tuple[str, str]] = []
        for name, server_config in self.config.servers.items():
            if not server_config.enabled:
                continue
            error = self._connect_one(name)
            if error is not None:
                failures.append((name, error))
        self.errors = list(failures)
        return failures

    def _connect_one(self, name: str) -> str | None:
        if name in self._clients:
            return None
        client = MCPClient(self.config.servers[name])
        try:
            self._call(client.connect(), timeout=self.config.servers[name].timeout)
        except MCPUnavailableError as e:
            error = str(e)
        except Exception as e:  # noqa: BLE001 - per-server failure, never fatal
            error = f"MCP server {name!r} failed to connect: {e}"
        else:
            self._clients[name] = client
            return None
        # A failed connect may still have spawned the server process /
        # opened an HTTP session before failing (e.g. timeout mid-
        # handshake) — close the client so nothing leaks.  Best effort:
        # teardown noise must never mask the original error.
        with contextlib.suppress(Exception):  # teardown noise
            self._call(client.close(), timeout=self.config.servers[name].timeout)
        return error

    def disconnect(self, name: str) -> None:
        """Disconnect one server and drop its discovered tools."""
        client = self._clients.pop(name, None)
        if client is None:
            return
        with contextlib.suppress(Exception):  # teardown noise
            self._call(client.close())
        self._discovered.discard(name)
        for key in [k for k in self._tools if k.startswith(name + "__")]:
            del self._tools[key]

    def close_all(self) -> None:
        """Disconnect every server and stop the event-loop thread."""
        for name in list(self._clients):
            self.disconnect(name)
        self._tools.clear()
        self._discovered.clear()
        if self._loop is not None:
            self._loop.close()
            self._loop = None

    # ------------------------------------------------------------------
    # tool discovery (once per session, not per turn)
    # ------------------------------------------------------------------
    def discover_tools(self) -> list[MCPToolSpec]:
        """tools/list each connected server; the result is cached.

        Called once at session start; a refresh requires reconnecting
        the server (``disconnect`` + ``connect_all``).  Returns the
        specs discovered by THIS call (newly discovered only), so a
        caller can register exactly what changed.
        """
        specs: list[MCPToolSpec] = []
        for name, client in self._clients.items():
            if name in self._discovered:
                continue
            self._discovered.add(name)
            try:
                raw_tools = self._call(
                    client.list_tools(), timeout=self.config.servers[name].timeout
                )
            except Exception as e:  # noqa: BLE001 - per-server failure, never fatal
                self.errors.append((name, f"MCP server {name!r} tool discovery failed: {e}"))
                continue
            for t in raw_tools:
                spec = MCPToolSpec(
                    server=name,
                    name=t["name"],
                    description=t.get("description") or "",
                    input_schema=t.get("input_schema") or {},
                )
                self._tools[f"{name}__{t['name']}"] = spec
                specs.append(spec)
        return specs

    def tool_specs(self) -> list[MCPToolSpec]:
        """All discovered specs, for building harness tools."""
        return list(self._tools.values())

    # ------------------------------------------------------------------
    # tool calls
    # ------------------------------------------------------------------
    def call_tool(
        self,
        server: str,
        tool: str,
        arguments: dict[str, Any],
        timeout: float | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Call TOOL on SERVER; returns the plain result dict from
        :meth:`MCPClient.call_tool`.

        Raises on connection/protocol errors (the tool adapter turns
        them into ``Error: ...`` strings); server-reported failures are
        flagged with ``is_error`` in the result dict, not raised.
        ``cancel_check`` (when given) is polled while waiting, so a
        Ctrl-C unblocks a hung call (see ``_LoopThread.run``).
        """
        client = self._clients.get(server)
        if client is None:
            raise ConnectionError(f"MCP server {server!r} is not connected")
        effective_timeout = client.config.timeout if timeout is None else timeout
        return self._call(
            client.call_tool(tool, arguments),
            timeout=effective_timeout,
            cancel_check=cancel_check,
        )
