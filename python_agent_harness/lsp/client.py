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
from pathlib import Path
from typing import Any


class LSPError(RuntimeError):
    pass


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

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None and not self._closed

    def start(self) -> None:
        if self.alive:
            return
        self._closed = False
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
        self._reader = threading.Thread(
            target=self._read_loop, daemon=True, name=f"lsp-reader:{self.command[0]}"
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
            kind, value = waiter.get(timeout=self.timeout)
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
        was_alive = proc is not None and proc.poll() is None
        if was_alive:
            with contextlib.suppress(Exception):
                self.request("shutdown", None)
            with contextlib.suppress(Exception):
                self.notify("exit", None)
        self._closed = True
        proc = self.proc
        if proc is not None and proc.poll() is None:
            with contextlib.suppress(Exception):
                proc.terminate()
                proc.wait(timeout=2)
            with contextlib.suppress(Exception):
                proc.kill()
        with self._state_lock:
            with contextlib.suppress(queue.Full):
                for waiter in self._pending.values():
                    waiter.put_nowait(("error", {"message": "LSP client closed"}))
            self._pending.clear()
            self._opened.clear()
            self._texts.clear()

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

    def _read_loop(self) -> None:
        try:
            proc = self.proc
            if proc is None or proc.stdout is None:
                return
            stream = proc.stdout
            while not self._closed:
                headers: dict[str, str] = {}
                while True:
                    line = stream.readline()
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
                payload = stream.read(length)
                if len(payload) != length:
                    return
                try:
                    message = json.loads(payload.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                self._handle_message(message)
        finally:
            with self._state_lock:
                pending = list(self._pending.values())
                self._pending.clear()
            with contextlib.suppress(queue.Full):
                for waiter in pending:
                    waiter.put_nowait(("error", {"message": "LSP server exited"}))

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
