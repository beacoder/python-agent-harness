"""Extra LSP client tests: lifecycle (start/close) and read-loop edges.

test_lsp.py's TestClientWireProtocol drives _send/_read_loop through
piped fakes but bypasses ``start()`` (it pre-seeds ``proc``).  These
tests cover the remaining lifecycle surface:

- ``start()``: spawn, initialize handshake, capability/encoding
  discovery, the "initialized" notification, failure paths (spawn
  OSError, initialize failure -> close + raise).
- ``close()``: the live-server path (shutdown request, exit
  notification, terminate), and draining pending waiters.
- ``_read_loop``: malformed framing (bad Content-Length, non-JSON
  payload, short payload) and unknown-id responses.
- ``_handle_message``: responses with unknown ids and server-initiated
  requests in the generic fallback branch.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from python_agent_harness.lsp import client as lsp_client_module
from python_agent_harness.lsp.client import LSPClient, LSPError


def _make_client(**kwargs) -> LSPClient:
    defaults = dict(command=["noop"], root="/tmp", language_id="python")
    defaults.update(kwargs)
    return LSPClient(**defaults)


def _read_framed(fd_read: int, timeout: float = 5.0):
    """Read one Content-Length framed JSON-RPC message from a pipe."""
    buf = b""
    deadline = time.monotonic() + timeout
    while b"\r\n\r\n" not in buf:
        self_deadline = deadline - time.monotonic()
        if self_deadline <= 0:
            raise AssertionError("timed out waiting for framed header")
        chunk = os.read(fd_read, 65536)
        if not chunk:
            return None
        buf += chunk
    header, _, rest = buf.partition(b"\r\n\r\n")
    length = int(header.split(b":")[1])
    while len(rest) < length:
        rest += os.read(fd_read, 65536)
    return json.loads(rest[:length])


def _write_framed(fd_write: int, message: dict) -> None:
    body = json.dumps(message).encode()
    os.write(fd_write, b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)


class TestStartLifecycle(unittest.TestCase):
    def _piped_proc(self):
        """Fake Popen whose streams are pipe ends the test controls."""
        stdin_r, stdin_w = os.pipe()
        stdout_r, stdout_w = os.pipe()
        proc = SimpleNamespace(
            stdin=os.fdopen(stdin_w, "wb", buffering=0),
            stdout=os.fdopen(stdout_r, "rb", buffering=0),
            poll=lambda: None,
            terminate=mock.Mock(),
            wait=mock.Mock(return_value=0),
            kill=mock.Mock(),
        )
        return proc, stdin_r, stdout_w

    def test_start_completes_initialize_handshake(self):
        client = _make_client(timeout=0.2)
        proc, stdin_r, stdout_w = self._piped_proc()
        failure: list[BaseException] = []

        def run() -> None:
            try:
                client.start()
            except BaseException as exc:  # noqa: BLE001 - asserted below
                failure.append(exc)

        with mock.patch.object(lsp_client_module.subprocess, "Popen", return_value=proc):
            t = threading.Thread(target=run)
            t.start()
            request = _read_framed(stdin_r)
            self.assertEqual(request["method"], "initialize")
            self.assertIn("capabilities", request["params"])
            _write_framed(
                stdout_w,
                {
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "result": {"capabilities": {"positionEncoding": "utf-8"}},
                },
            )
            initialized = _read_framed(stdin_r)
            t.join(timeout=5)
        self.assertEqual(failure, [])
        self.assertEqual(initialized["method"], "initialized")
        self.assertEqual(client.position_encoding, "utf-8")
        self.assertTrue(client.alive)
        # Release the reader before close(): closing a stream that has a
        # concurrent blocked read waits for that read to finish, and the
        # mocked process never dies to break the pipe.  Closing the
        # server's write end gives the reader EOF; join it, then close
        # the client (no pending read -> no wait).
        os.close(stdout_w)
        client._reader.join(timeout=5)
        client.close()
        os.close(stdin_r)

    def test_start_defaults_encoding_when_server_silent(self):
        client = _make_client(timeout=0.2)
        proc, stdin_r, stdout_w = self._piped_proc()

        def run() -> None:
            with contextlib.suppress(Exception):
                client.start()

        with mock.patch.object(lsp_client_module.subprocess, "Popen", return_value=proc):
            t = threading.Thread(target=run)
            t.start()
            request = _read_framed(stdin_r)
            # capabilities without positionEncoding: stays utf-16
            _write_framed(stdout_w, {"jsonrpc": "2.0", "id": request["id"], "result": {}})
            _read_framed(stdin_r)  # initialized
            t.join(timeout=5)
        self.assertEqual(client.position_encoding, "utf-16")
        # same hazard as above: EOF the reader, join, then close
        os.close(stdout_w)
        client._reader.join(timeout=5)
        client.close()
        os.close(stdin_r)

    def test_start_oserror_raises_lsp_error(self):
        client = _make_client()
        with (
            mock.patch.object(
                lsp_client_module.subprocess, "Popen", side_effect=OSError("no such binary")
            ),
            self.assertRaises(LSPError) as ctx,
        ):
            client.start()
        self.assertIn("failed to start LSP server", str(ctx.exception))

    def test_start_failure_during_initialize_closes_and_raises(self):
        client = _make_client()
        # close() runs inside start()'s failure path, so the reader must
        # unblock on its own: an EOF-returning dummy stdout (and a write
        # sink stdin) avoid the "close waits for a pending read" hazard
        # without needing to pre-release anything.
        eof_stream = SimpleNamespace(readline=lambda: b"", close=lambda: None)
        sink = SimpleNamespace(write=lambda b: None, flush=lambda: None, close=lambda: None)
        proc = SimpleNamespace(
            stdin=sink,
            stdout=eof_stream,
            poll=lambda: None,
            terminate=mock.Mock(),
            wait=mock.Mock(return_value=0),
            kill=mock.Mock(),
        )
        with (
            mock.patch.object(lsp_client_module.subprocess, "Popen", return_value=proc),
            mock.patch.object(LSPClient, "request", side_effect=LSPError("handshake failed")),
            self.assertRaises(LSPError),
        ):
            client.start()
        self.assertTrue(client._closed)
        client._reader.join(timeout=5)  # reader saw EOF and exited

    def test_start_when_already_alive_is_noop(self):
        client = _make_client()
        client.proc = SimpleNamespace(poll=lambda: None)
        client._closed = False
        with mock.patch.object(lsp_client_module.subprocess, "Popen") as popen:
            client.start()
        popen.assert_not_called()


class TestRequestAndNotifyEdges(unittest.TestCase):
    def test_request_send_failure_drops_pending_entry(self):
        """When _send fails mid-request, the pending waiter must be
        removed — otherwise it leaks forever in _pending."""
        client = _make_client()
        client.start_if_needed = lambda: None
        with (
            mock.patch.object(client, "_send", side_effect=LSPError("server down")),
            self.assertRaises(LSPError),
        ):
            client.request("textDocument/hover")
        self.assertEqual(client._pending, {})

    def test_request_without_params_omits_params_key(self):
        sent: list[dict] = []
        client = _make_client(timeout=0.05)
        client.start_if_needed = lambda: None
        client._send = sent.append
        with self.assertRaises(LSPError):  # no response -> timeout
            client.request("shutdown")
        self.assertEqual(sent[0]["method"], "shutdown")
        self.assertNotIn("params", sent[0])

    def test_notify_without_params_omits_params_key(self):
        sent: list[dict] = []
        client = _make_client()
        client.start_if_needed = lambda: None
        client._send = sent.append
        client.notify("exit")
        self.assertEqual(sent[0], {"jsonrpc": "2.0", "method": "exit"})

    def test_notify_starts_server_if_needed(self):
        client = _make_client()
        started: list[bool] = []
        client.start_if_needed = lambda: started.append(True)
        client._send = lambda m: None  # type: ignore[method-assign]
        client.notify("ping")
        self.assertEqual(started, [True])


class TestCloseLifecycle(unittest.TestCase):
    def test_close_live_server_sends_shutdown_and_terminates(self):
        packets: list[bytes] = []
        proc = SimpleNamespace(
            stdin=SimpleNamespace(write=packets.append, flush=lambda: None, close=lambda: None),
            stdout=SimpleNamespace(close=lambda: None),
            poll=lambda: None,
            terminate=mock.Mock(),
            wait=mock.Mock(return_value=0),
            kill=mock.Mock(),
        )
        client = _make_client(timeout=0.05)
        client.proc = proc
        client._closed = False
        client.close()
        methods = [json.loads(p.split(b"\r\n\r\n", 1)[1])["method"] for p in packets]
        self.assertIn("shutdown", methods)
        self.assertIn("exit", methods)
        proc.terminate.assert_called_once()
        self.assertTrue(client._closed)

    def test_close_drains_pending_waiters_with_error(self):
        client = _make_client()
        client.proc = None
        waiter: queue.Queue = queue.Queue(maxsize=1)
        client._pending[77] = waiter
        client.close()
        self.assertEqual(waiter.get(timeout=1), ("error", {"message": "LSP client closed"}))

    def test_alive_property_states(self):
        client = _make_client()
        self.assertFalse(client.alive)  # no proc
        client.proc = SimpleNamespace(poll=lambda: 1)
        self.assertFalse(client.alive)  # exited
        client.proc = SimpleNamespace(poll=lambda: None)
        client._closed = True
        self.assertFalse(client.alive)  # closed
        client._closed = False
        self.assertTrue(client.alive)


class TestReadLoopEdges(unittest.TestCase):
    def _run_with_bytes(self, data: bytes) -> LSPClient:
        client = _make_client()
        client.start_if_needed = lambda: None
        client._closed = False
        r, w = os.pipe()
        os.write(w, data)
        os.close(w)
        stream = os.fdopen(r, "rb", buffering=0)
        client.proc = SimpleNamespace(stdout=stream)
        try:
            client._read_loop()  # returns at EOF
        finally:
            stream.close()
        return client

    def test_bad_content_length_is_skipped(self):
        self._run_with_bytes(b"Content-Length: nope\r\n\r\n")

    def test_invalid_json_payload_is_skipped(self):
        self._run_with_bytes(b"Content-Length: 5\r\n\r\n{{{{{")

    def test_short_payload_ends_the_loop(self):
        self._run_with_bytes(b"Content-Length: 100\r\n\r\nshort")

    def test_header_line_without_colon_is_ignored(self):
        self._run_with_bytes(b"GARBAGE\r\nContent-Length: 2\r\n\r\n{}")

    def test_unknown_response_id_is_ignored(self):
        body = json.dumps({"jsonrpc": "2.0", "id": 999, "result": 1}).encode()
        client = self._run_with_bytes(
            b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
        )
        self.assertEqual(client._pending, {})

    def test_response_with_non_int_id_ignored(self):
        client = _make_client()
        client._handle_message({"jsonrpc": "2.0", "id": "abc", "result": 1})
        self.assertEqual(client._pending, {})

    def test_diagnostics_returns_a_copy(self):
        client = _make_client()
        client._diagnostics["file:///x.py"] = [{"message": "m"}]
        got = client.diagnostics("file:///x.py")
        got.append({"message": "new"})
        self.assertEqual(client.diagnostics("file:///x.py"), [{"message": "m"}])

    def test_server_request_fallback_replies_null(self):
        packets: list[bytes] = []
        client = _make_client()
        client.proc = SimpleNamespace(
            stdin=SimpleNamespace(write=packets.append, flush=lambda: None),
            poll=lambda: None,
        )
        client._handle_message({"id": 4, "method": "custom/request", "params": {}})
        self.assertEqual(
            json.loads(packets[0].split(b"\r\n\r\n", 1)[1]),
            {"jsonrpc": "2.0", "id": 4, "result": None},
        )

    def test_server_request_reply_failure_is_suppressed(self):
        client = _make_client()
        client.proc = None  # _send raises LSPError; must be swallowed
        client._handle_message({"id": 5, "method": "workspace/configuration", "params": []})


if __name__ == "__main__":
    unittest.main()
