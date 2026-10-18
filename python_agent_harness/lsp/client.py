"""Small, dependency-free LSP/JSON-RPC client used by the agent harness.

This intentionally implements only the transport/lifecycle pieces needed by the
agent-facing LSP tool. It is synchronous at the public boundary because the
harness' Tool API is synchronous; a reader thread keeps server messages flowing.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


class LSPError(RuntimeError):
    pass


# Upper bound for one framed JSON-RPC message.  A Content-Length header is
# untrusted input: a bogus or malicious value must not make the reader
# reserve gigabytes.  Real LSP responses (even documentSymbol on a huge
# file) are orders of magnitude smaller.
MAX_MESSAGE_BYTES = 64 * 1024 * 1024


class LSPClient:
    def __init__(self, command: list[str], root: str, language_id: str, timeout: float = 30.0):
        self.command = command
        self.root = os.path.realpath(root)
        self.language_id = language_id
        self.timeout = timeout
        self.proc: subprocess.Popen[bytes] | None = None
        self._reader: threading.Thread | None = None
        self._write_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._next_id = 1
        self._pending: dict[int, queue.Queue] = {}
        self._opened: dict[str, int] = {}
        self._texts: dict[str, str] = {}
        self._diagnostics: dict[str, list[dict[str, Any]]] = {}
        self._capabilities: dict[str, Any] = {}
        self.position_encoding = "utf-16"
        self._closed = False
        # Set when the reader thread exits for ANY reason other than an
        # explicit close(): the transport can no longer deliver responses,
        # so the client must count as dead even while the child process is
        # still running (a framing error, a crashed/hung server).  Without
        # this, `alive` kept reporting True and `manager.get_client` never
        # replaced the broken client -- every later LSP call then burned
        # the full request timeout for the rest of the session.
        self._reader_dead = False
        # Identity of the CURRENT reader thread's generation.  A reader
        # superseded by start() can finish after the new one is installed;
        # without this counter it would mark the fresh (healthy) transport
        # dead in its finally block -- the proc identity check alone does
        # not cover the window before the new proc is assigned.
        self._reader_gen = 0

    @property
    def alive(self) -> bool:
        return (
            self.proc is not None
            and self.proc.poll() is None
            and not self._closed
            and not self._reader_dead
        )

    def start(self) -> None:
        if self.alive:
            return
        # Invalidate any superseded reader FIRST: _release_proc() below
        # unblocks it, and if it reached its finally block while the
        # generation still matched it would mark this fresh transport dead
        # (and fail its waiters) before the replacement is installed.
        self._reader_gen += 1
        # A previous child can still be running when only the reader died
        # (framing error / crashed transport): release it before the pipes
        # are replaced, or the process and its descriptors leak.
        if self.proc is not None:
            self._release_proc()
        self._closed = False
        self._reader_dead = False
        # A fresh server has no documents open: forget the previous
        # session's bookkeeping, or open_document() would send didChange
        # for a document this process never opened.
        with self._state_lock:
            self._opened.clear()
            self._texts.clear()
        self.position_encoding = "utf-16"
        try:
            self.proc = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                cwd=self.root,
                bufsize=0,
            )
        except OSError as e:
            raise LSPError(f"failed to start LSP server {' '.join(self.command)!r}: {e}") from e
        gen = self._reader_gen
        self._reader = threading.Thread(
            target=self._read_loop, args=(gen,), daemon=True, name=f"lsp-reader:{self.command[0]}"
        )
        self._reader.start()
        try:
            result = self.request(
                "initialize",
                {
                    "processId": os.getpid(),
                    "rootUri": Path(self.root).as_uri(),
                    "workspaceFolders": [
                        {"uri": Path(self.root).as_uri(), "name": Path(self.root).name}
                    ],
                    "capabilities": {
                        "general": {"positionEncodings": ["utf-16", "utf-8"]},
                        "workspace": {
                            "workspaceFolders": True,
                            "symbol": {"dynamicRegistration": False},
                        },
                        "textDocument": {
                            "definition": {"linkSupport": True},
                            "references": {},
                            "hover": {"contentFormat": ["markdown", "plaintext"]},
                            "documentSymbol": {"hierarchicalDocumentSymbolSupport": True},
                            "implementation": {"linkSupport": True},
                            "callHierarchy": {"dynamicRegistration": False},
                        },
                    },
                    "trace": "off",
                },
            )
            self._capabilities = result.get("capabilities", {}) if isinstance(result, dict) else {}
            encoding = self._capabilities.get("positionEncoding")
            if encoding in {"utf-8", "utf-16", "utf-32"}:
                self.position_encoding = encoding
            self.notify("initialized", {})
        except Exception:
            self.close()
            raise

    def request(self, method: str, params: Any = None) -> Any:
        self.start_if_needed()
        with self._state_lock:
            request_id = self._next_id
            self._next_id += 1
            waiter: queue.Queue = queue.Queue(maxsize=1)
            self._pending[request_id] = waiter
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        try:
            self._send(message)
            kind, value = self._wait_response(waiter, method)
        except queue.Empty as e:
            with self._state_lock:
                self._pending.pop(request_id, None)
            raise LSPError(f"LSP request timed out: {method}") from e
        except BaseException:
            # _send() may raise (server down / write failure); drop the
            # now-orphaned pending entry so it does not leak forever.
            with self._state_lock:
                self._pending.pop(request_id, None)
            raise
        if kind == "error":
            code = value.get("code") if isinstance(value, dict) else None
            msg = (
                value.get("message", "unknown LSP error") if isinstance(value, dict) else str(value)
            )
            raise LSPError(f"LSP {method} failed ({code}): {msg}")
        return value

    def _wait_response(self, waiter: queue.Queue, method: str) -> tuple[str, Any]:
        """Wait for WAITER, failing fast once the transport is dead.

        The reader drains every waiter when it exits, but a request that
        registers AFTER that drain finds an empty pending map and could
        only wait out the full request timeout for a response nobody can
        deliver.  Polling in slices lets the wait end as soon as the
        transport is known to be dead.  Raises ``queue.Empty`` on a real
        timeout so the caller keeps its single timeout message.
        """
        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise queue.Empty
            try:
                return waiter.get(timeout=min(0.1, remaining))
            except queue.Empty:
                if self._reader_dead:
                    raise LSPError(f"LSP server exited while waiting for {method}") from None

    def notify(self, method: str, params: Any = None) -> None:
        self.start_if_needed()
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        self._send(message)

    def start_if_needed(self) -> None:
        if not self.alive:
            self.start()

    def open_document(self, uri: str, text: str) -> None:
        self.start_if_needed()
        # Decide which notification to send while holding _state_lock, but
        # send it AFTER releasing the lock: notify()->_send() does a blocking
        # write on the server's stdin pipe, and holding _state_lock across that
        # write can deadlock the reader thread (which needs _state_lock to
        # deliver responses/diagnostics) if the pipe buffer fills.
        with self._state_lock:
            gen = self._reader_gen
            version = self._opened.get(uri, 0)
            if version:
                if self._texts.get(uri) == text:
                    return
                version += 1
                self._opened[uri] = version
                self._texts[uri] = text
                method = "textDocument/didChange"
                params: dict[str, Any] = {
                    "textDocument": {"uri": uri, "version": version},
                    "contentChanges": [{"text": text}],
                }
            else:
                self._opened[uri] = 1
                self._texts[uri] = text
                method = "textDocument/didOpen"
                params = {
                    "textDocument": {
                        "uri": uri,
                        "languageId": self.language_id,
                        "version": 1,
                        "text": text,
                    }
                }
        self.notify(method, params)
        if self._reader_gen != gen:
            # The transport restarted between the bookkeeping above and the
            # send: start() cleared _opened, and this notification either
            # went to a server that is gone or to a replacement that never
            # received didOpen.  Drop the stale claim so the next call
            # re-opens the document against the current server instead of
            # sending didChange for a document it does not know.
            with self._state_lock:
                self._opened.pop(uri, None)
                self._texts.pop(uri, None)

    def close_document(self, uri: str) -> None:
        with self._state_lock:
            if uri not in self._opened:
                return
            self._opened.pop(uri, None)
            self._texts.pop(uri, None)
        if self.alive:
            self.notify("textDocument/didClose", {"textDocument": {"uri": uri}})

    def diagnostics(self, uri: str) -> list[dict[str, Any]]:
        with self._state_lock:
            return list(self._diagnostics.get(uri, []))

    def close(self) -> None:
        if self._closed:
            return
        proc = self.proc
        # The graceful path (shutdown request + exit notification) goes
        # through request()/notify(), which call start_if_needed() — and
        # with the reader dead `alive` is False, so closing would SPAWN A
        # FRESH SERVER just to shut it down, leaking the old process.
        # When the transport is already broken there is nobody to talk to:
        # go straight to terminate.
        was_alive = proc is not None and proc.poll() is None and not self._reader_dead
        if was_alive:
            with contextlib.suppress(Exception):
                self.request("shutdown", None)
            with contextlib.suppress(Exception):
                self.notify("exit", None)
        self._closed = True
        self._release_proc()
        with self._state_lock:
            with contextlib.suppress(queue.Full):
                for waiter in self._pending.values():
                    waiter.put_nowait(("error", {"message": "LSP client closed"}))
            self._pending.clear()
            self._opened.clear()
            self._texts.clear()

    def _release_proc(self) -> None:
        """Terminate + reap the current server process and close its pipes.

        Shared by ``close()`` (explicit teardown) and ``start()`` (a
        restart after the reader died): in the latter case the child may
        still be running, and the pipes MUST be released either way --
        ``subprocess`` does not close them on terminate/kill, so leaving
        them open leaks descriptors and emits ResourceWarning.
        """
        proc = self.proc
        if proc is None:
            return
        if proc.poll() is None:
            with contextlib.suppress(Exception):
                proc.terminate()
                proc.wait(timeout=2)
            with contextlib.suppress(Exception):
                proc.kill()
            # Reap the killed child so it does not linger as a zombie
            # until subprocess's global cleanup runs.
            with contextlib.suppress(Exception):
                proc.wait(timeout=2)
        for stream in (getattr(proc, "stdin", None), getattr(proc, "stdout", None)):
            if stream is not None:
                with contextlib.suppress(Exception):
                    stream.close()

    def _send(self, message: dict[str, Any]) -> None:
        data = json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        packet = b"Content-Length: " + str(len(data)).encode("ascii") + b"\r\n\r\n" + data
        proc = self.proc
        if proc is None or proc.stdin is None or proc.poll() is not None:
            raise LSPError("LSP server is not running")
        with self._write_lock:
            try:
                proc.stdin.write(packet)
                proc.stdin.flush()
            except OSError as e:
                raise LSPError(f"failed writing to LSP server: {e}") from e

    def _read_loop(self, gen: int | None = None) -> None:
        proc = self.proc
        try:
            if proc is None or proc.stdout is None:
                return
            stream = proc.stdout
            while not self._closed:
                headers: dict[str, str] = {}
                while True:
                    try:
                        line = stream.readline()
                    except (ValueError, OSError):
                        # close() may release the pipes while this thread
                        # is parked in a read; that surfaces here as a
                        # closed-file/most-unblocked error.  Treat it as
                        # EOF instead of letting a traceback escape a
                        # daemon thread.
                        return
                    if not line:
                        return
                    line = line.rstrip(b"\r\n")
                    if not line:
                        break
                    if b":" in line:
                        key, value = line.split(b":", 1)
                        headers[key.decode("ascii", "replace").lower()] = value.strip().decode(
                            "ascii", "replace"
                        )
                try:
                    length = int(headers.get("content-length", "0"))
                except ValueError:
                    continue
                if length < 0 or length > MAX_MESSAGE_BYTES:
                    # Unusable framing: a bogus/huge header must not make
                    # read() reserve gigabytes, and a negative one can
                    # never be satisfied.  Stop reading; the finally block
                    # marks the transport dead so the server is replaced.
                    return
                payload = self._read_exactly(stream, length)
                if payload is None:
                    return
                try:
                    message = json.loads(payload.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                self._handle_message(message)
        finally:
            # Mark the transport dead for every exit path except an
            # explicit close(): `alive` must go False so the manager
            # replaces this client instead of sending requests into a
            # pipe nobody reads (which only produced timeouts).  Both
            # guards matter: a superseded reader (start() installed a
            # replacement) must not kill the fresh transport, and a
            # concurrent close() already owns the flag.
            current = (gen is None or gen == self._reader_gen) and self.proc is proc
            if current:
                self._reader_dead = True
                # Only the CURRENT reader owns this client's waiters: a
                # superseded reader exiting during a restart must not fail
                # requests that belong to the replacement transport.
                with self._state_lock:
                    pending = list(self._pending.values())
                    self._pending.clear()
                for waiter in pending:
                    # Per waiter: one already-filled queue must not stop
                    # the rest from being notified.
                    with contextlib.suppress(queue.Full):
                        waiter.put_nowait(("error", {"message": "LSP server exited"}))

    @staticmethod
    def _read_exactly(stream: Any, length: int) -> bytes | None:
        """Read exactly LENGTH bytes from STREAM; None on EOF first.

        ``read(n)`` on a pipe-backed raw stream returns UP TO n bytes: it
        reports what a single ``read(2)`` delivered, which for a message
        larger than the pipe buffer is a partial message.  Treating that
        short read as EOF killed the reader thread permanently -- one
        large documentSymbol/hover/findReferences result silently
        disabled LSP for the rest of the session.  Accumulate until the
        framed length is satisfied; only a genuinely empty read (every
        writer gone) is EOF.
        """
        if length == 0:
            return b""
        chunks: list[bytes] = []
        remaining = length
        while remaining:
            try:
                chunk = stream.read(remaining)
            except (ValueError, OSError):
                return None  # pipe closed under us (see the readline guard)
            if not chunk:
                return None
            chunks.append(chunk)
            # Clamp: a stream that over-delivers must not make `remaining`
            # negative (a negative read() means "until EOF", i.e. a loop).
            remaining = max(0, remaining - len(chunk))
        return b"".join(chunks)

    def _handle_message(self, message: dict[str, Any]) -> None:
        if "id" in message and ("result" in message or "error" in message):
            request_id = message.get("id")
            if isinstance(request_id, int):
                with self._state_lock:
                    waiter = self._pending.pop(request_id, None)
                if waiter is not None:
                    waiter.put(
                        ("error", message["error"])
                        if "error" in message
                        else ("result", message.get("result"))
                    )
            return

        method = message.get("method")
        params = message.get("params")
        if method == "textDocument/publishDiagnostics" and isinstance(params, dict):
            uri = params.get("uri")
            if isinstance(uri, str):
                with self._state_lock:
                    self._diagnostics[uri] = params.get("diagnostics", []) or []
            return

        # Servers may send requests/notifications to the client. Respond to
        # common client-side requests so a server does not stall waiting for
        # configuration/progress/UI handling that the coding agent does not need.
        if "id" in message and isinstance(message.get("id"), int):
            method = str(method or "")
            if method == "workspace/configuration":
                items = params if isinstance(params, list) else []
                result = [None for _ in items]
            elif method == "workspace/applyEdit":
                result = {"applied": False}
            else:
                result = None
            with contextlib.suppress(LSPError):
                self._send({"jsonrpc": "2.0", "id": message["id"], "result": result})
