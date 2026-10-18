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


class TestReadLoopFraming(unittest.TestCase):
    """The reader must reassemble a message split across several reads.

    ``Popen(bufsize=0)`` gives a raw ``FileIO``, whose ``read(n)`` returns
    whatever one ``read(2)`` delivered -- fewer than n bytes for anything
    larger than what is buffered.  Treating that short read as EOF killed
    the reader thread permanently: one large documentSymbol/hover/
    findReferences result silently disabled LSP for the rest of the
    session (``alive`` kept reporting True, so the manager never replaced
    the client).
    """

    def _client_with_stream(self, stream) -> LSPClient:
        client = _make_client(timeout=2)
        client.start_if_needed = lambda: None
        client._closed = False
        client.proc = SimpleNamespace(stdout=stream, poll=lambda: None)
        return client

    def test_message_split_across_reads_is_reassembled(self):
        body = json.dumps({"jsonrpc": "2.0", "id": 7, "result": {"ok": True}}).encode()
        header = b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n"
        r, w = os.pipe()

        # A delayed, chunked writer: the first read(2) can only deliver a
        # fragment, which is exactly what the old code treated as EOF.
        def write_slowly() -> None:
            os.write(w, header)
            time.sleep(0.05)
            os.write(w, body[:4])
            time.sleep(0.05)
            os.write(w, body[4:])
            os.close(w)

        writer = threading.Thread(target=write_slowly, daemon=True)
        writer.start()
        stream = os.fdopen(r, "rb", buffering=0)
        client = self._client_with_stream(stream)
        waiter: queue.Queue = queue.Queue(maxsize=1)
        client._pending[7] = waiter
        try:
            client._read_loop()
        finally:
            stream.close()
            writer.join(timeout=5)
        self.assertEqual(waiter.get(timeout=5), ("result", {"ok": True}))

    def test_reader_exit_marks_the_transport_dead(self):
        """alive must go False so the manager replaces the client."""
        r, w = os.pipe()
        os.write(w, b"")
        os.close(w)  # immediate EOF
        stream = os.fdopen(r, "rb", buffering=0)
        client = self._client_with_stream(stream)
        self.assertTrue(client.alive)
        try:
            client._read_loop()
        finally:
            stream.close()
        self.assertFalse(client.alive)
        self.assertTrue(client._reader_dead)

    def test_dead_reader_is_replaced_by_start(self):
        """start() clears the dead flag and the stale document bookkeeping."""
        client = _make_client()
        client._reader_dead = True
        client._opened["file:///x"] = 3
        client._texts["file:///x"] = "old"
        proc = SimpleNamespace(
            stdin=SimpleNamespace(write=lambda b: None, flush=lambda: None, close=lambda: None),
            stdout=SimpleNamespace(readline=lambda: b""),
            poll=lambda: None,
            terminate=mock.Mock(),
            wait=mock.Mock(return_value=0),
            kill=mock.Mock(),
        )
        with (
            mock.patch.object(lsp_client_module.subprocess, "Popen", return_value=proc),
            contextlib.suppress(Exception),
        ):
            client.start()
        self.assertFalse(client._reader_dead)
        self.assertEqual(client._opened, {})
        self.assertEqual(client._texts, {})
        client._reader.join(timeout=5)

    def test_oversized_content_length_stops_the_reader(self):
        """A bogus/huge header must not reserve gigabytes or spin."""
        r, w = os.pipe()
        os.write(w, b"Content-Length: 999999999\r\n\r\n")
        os.close(w)
        stream = os.fdopen(r, "rb", buffering=0)
        client = self._client_with_stream(stream)
        try:
            client._read_loop()  # returns instead of trying to read 1 GB
        finally:
            stream.close()
        self.assertFalse(client.alive)


class TestCloseAfterReaderDeath(unittest.TestCase):
    """close() must tear down, never RESTART, a client whose reader died.

    ``alive`` is False once the reader exits, and close()'s graceful path
    goes through request()/notify(), both of which call
    ``start_if_needed()``.  Without a guard, closing a client whose
    transport broke while the child was still running spawned a brand-new
    server just to shut it down -- leaking the old process and paying a
    full initialize handshake on every teardown.
    """

    def test_close_does_not_spawn_a_replacement_server(self):
        client = _make_client(timeout=0.2)
        proc = SimpleNamespace(
            stdin=SimpleNamespace(write=lambda b: None, flush=lambda: None, close=lambda: None),
            stdout=SimpleNamespace(readline=lambda: b""),
            poll=lambda: None,
            terminate=mock.Mock(),
            wait=mock.Mock(return_value=0),
            kill=mock.Mock(),
        )
        client.proc = proc
        client._closed = False
        client._reader_dead = True  # transport already broken
        with (
            mock.patch.object(lsp_client_module.subprocess, "Popen") as popen,
            mock.patch.object(LSPClient, "request") as request,
        ):
            client.close()
        popen.assert_not_called()
        request.assert_not_called()
        self.assertTrue(client._closed)
        proc.terminate.assert_called()

    def test_close_still_shuts_down_gracefully_when_healthy(self):
        client = _make_client(timeout=0.2)
        client.proc = SimpleNamespace(
            stdin=SimpleNamespace(write=lambda b: None, flush=lambda: None, close=lambda: None),
            stdout=SimpleNamespace(readline=lambda: b""),
            poll=lambda: None,
            terminate=mock.Mock(),
            wait=mock.Mock(return_value=0),
            kill=mock.Mock(),
        )
        client._closed = False
        client._reader_dead = False
        with mock.patch.object(LSPClient, "request") as request:
            client.close()
        request.assert_called_once_with("shutdown", None)


class TestReaderSurvivesClosedStream(unittest.TestCase):
    """A pipe closed underneath the reader is EOF, not a traceback."""

    def test_readline_on_closed_stream_exits_cleanly(self):
        r, w = os.pipe()
        os.close(w)
        stream = os.fdopen(r, "rb", buffering=0)
        stream.close()  # reader will hit "read of closed file"
        client = _make_client(timeout=1)
        client.start_if_needed = lambda: None
        client._closed = False
        client.proc = SimpleNamespace(stdout=stream, poll=lambda: None)
        client._read_loop()  # must not raise
        self.assertTrue(client._reader_dead)
        self.assertFalse(client.alive)

    def test_short_body_then_eof_is_eof_not_a_hang(self):
        """A truncated message at EOF ends the loop (no infinite retry)."""
        r, w = os.pipe()
        os.write(w, b"Content-Length: 50\r\n\r\nshort")
        os.close(w)
        stream = os.fdopen(r, "rb", buffering=0)
        client = _make_client(timeout=1)
        client.start_if_needed = lambda: None
        client._closed = False
        client.proc = SimpleNamespace(stdout=stream, poll=lambda: None)
        try:
            client._read_loop()
        finally:
            stream.close()
        self.assertFalse(client.alive)


class TestSupersededReader(unittest.TestCase):
    """A reader superseded by start() must not kill the fresh transport.

    start() releases the old process (closing its pipes) and installs a
    new one.  The old reader thread can observe the closed pipe AFTER the
    new process is installed; without the generation guard its finally
    block marked the healthy new client dead, so the very next request
    would fail with "server exited" and the manager would restart again --
    an endless restart loop under transient framing errors.
    """

    def test_stale_reader_does_not_mark_new_transport_dead(self):
        client = _make_client(timeout=1)
        client._closed = False
        client._reader_dead = False
        client._reader_gen = 2  # a replacement reader is current
        old_proc = SimpleNamespace(poll=lambda: None, stdout=SimpleNamespace(readline=lambda: b""))
        client.proc = old_proc
        # The superseded reader exits while holding the OLD generation.
        client._read_loop(gen=1)
        self.assertFalse(client._reader_dead)
        self.assertTrue(client.alive)

    def test_current_reader_still_marks_transport_dead(self):
        client = _make_client(timeout=1)
        client._closed = False
        client._reader_dead = False
        client._reader_gen = 3
        client.proc = SimpleNamespace(stdout=None, poll=lambda: None)
        client._read_loop(gen=3)
        self.assertTrue(client._reader_dead)
        self.assertFalse(client.alive)

    def test_stale_reader_does_not_drain_new_transport_waiters(self):
        """The drain must belong to the current reader only.

        A request registered on the replacement transport was failed with
        "LSP server exited" by the OLD reader's finally block -- the new
        connection was healthy, yet its in-flight request died and the
        caller saw the dead server's error.
        """
        client = _make_client(timeout=1)
        client._closed = False
        client._reader_dead = False
        client._reader_gen = 5
        stale_proc = SimpleNamespace(poll=lambda: 0, stdout=SimpleNamespace(readline=lambda: b""))
        client.proc = stale_proc
        fresh: queue.Queue = queue.Queue(maxsize=1)
        client._pending[9001] = fresh

        client._read_loop(gen=4)  # superseded reader exits

        self.assertTrue(fresh.empty(), "stale reader failed a fresh request")
        self.assertIn(9001, client._pending)

    def test_drain_notifies_every_waiter_when_one_queue_is_full(self):
        """One already-filled waiter must not stop the others being told."""
        client = _make_client(timeout=1)
        client._closed = False
        client._reader_dead = False
        client._reader_gen = 1
        client.proc = SimpleNamespace(stdout=None, poll=lambda: None)
        full: queue.Queue = queue.Queue(maxsize=1)
        full.put_nowait(("error", {"message": "already delivered"}))
        other: queue.Queue = queue.Queue(maxsize=1)
        client._pending[1] = full
        client._pending[2] = other
        client._read_loop(gen=1)
        self.assertEqual(other.get_nowait(), ("error", {"message": "LSP server exited"}))


class TestDocumentBookkeepingAcrossRestart(unittest.TestCase):
    """A restart must never leave a stale "this document is open" claim.

    start() clears _opened because a replacement server has no documents
    open.  If the restart lands between open_document()'s bookkeeping and
    its notification, the map would claim a document the new server never
    saw -- so the NEXT open_document() would send didChange instead of
    didOpen and the server would reject it.
    """

    def test_restart_during_open_clears_the_stale_claim(self):
        client = _make_client(timeout=1)
        client._closed = False
        client._reader_dead = False
        sent: list[str] = []

        def fake_notify(method, params=None):
            sent.append(method)
            # Simulate the transport dying and being replaced mid-send.
            client._reader_gen += 1
            client._opened.clear()
            client._texts.clear()

        client.notify = fake_notify
        client.start_if_needed = lambda: None
        client.open_document("file:///x.py", "x = 1\n")

        self.assertEqual(sent, ["textDocument/didOpen"])
        self.assertNotIn("file:///x.py", client._opened)
        self.assertNotIn("file:///x.py", client._texts)

    def test_reopening_after_restart_sends_did_open_again(self):
        client = _make_client(timeout=1)
        client._closed = False
        client._reader_dead = False
        sent: list[str] = []
        client.notify = lambda method, params=None: sent.append(method)
        client.start_if_needed = lambda: None

        client.open_document("file:///y.py", "y = 1\n")
        client.open_document("file:///y.py", "y = 2\n")  # same server: didChange
        client._opened.clear()  # as start() would
        client._texts.clear()
        client.open_document("file:///y.py", "y = 2\n")  # new server: didOpen

        self.assertEqual(
            sent,
            ["textDocument/didOpen", "textDocument/didChange", "textDocument/didOpen"],
        )
