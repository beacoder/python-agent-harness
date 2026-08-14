"""Concurrent sub-agent isolation: a sub-agent's connection drop,
retry, abort or stream must never affect a sibling's request.

Regression suite for the shared-Client race (review items 8/9/10).
Before the fix every sub-agent shared ONE httpx Client, so one
sub-agent's connection failure (``_reset_http`` swaps and closes the
shared pool), abort (the shared ``_aborted`` flag) or retry could tear
down a sibling's in-flight request.  Each invocation now runs on its
own Client clone (``AgentSession.run_subagent``).

All tests run concurrent sub-agents against a REAL HTTP server that
scripts per-request behaviors (success / 500-then-retry /
disconnect-mid-stream / slow-stream-for-cancel) and records every
request server-side.  The final response of each sub-agent must be
exactly its own content — no cross-contamination, no duplication (a
partial stream is discarded on retry), no loss, no error pollution —
and the server-side request counts pin the isolation: a healthy
sibling is never retried, a cancelled sub-agent stops promptly.
"""

from __future__ import annotations

import contextlib
import json
import re
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from python_agent_harness.agent_session import AgentSession
from python_agent_harness.client import Client
from python_agent_harness.tools import default_registry

SUCCESS_CHUNKS = 4
CHUNK_INTERVAL = 0.05


class ServerState:
    """Per-request bookkeeping shared with the HTTP handler."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.counts: dict[str, int] = {}
        self.statuses: dict[str, list[int]] = {}
        self.stream_flags: dict[str, list[bool]] = {}
        # set once the first chunk of a response has been flushed
        self.first_chunk: dict[str, threading.Event] = {}


class _Handler(BaseHTTPRequestHandler):
    """Behavior is scripted via markers in the sub-agent prompt:

    - BEHAVIOR:success    complete streaming response (N chunks)
    - BEHAVIOR:retry500   first two attempts answer HTTP 500, then success
    - BEHAVIOR:disconnect first attempt streams one chunk then drops the
                         connection mid-stream, later attempts succeed
    - BEHAVIOR:slow       long slow stream (aborted by the client)
    """

    protocol_version = "HTTP/1.1"

    def do_POST(self):
        state: ServerState = self.server.state
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        content = ""
        for m in body.get("messages") or []:
            if m.get("role") == "user":
                content = str(m.get("content") or "")
                break
        bm = re.search(r"BEHAVIOR:(\w+)", content)
        im = re.search(r"ID:(\w+)", content)
        hm = re.search(r"HOLD:([0-9.]+)", content)
        nm = re.search(r"N:(\d+)", content)
        behavior = bm.group(1) if bm else "success"
        sid = im.group(1) if im else "0"
        hold = float(hm.group(1)) if hm else 0.05
        nchunks = int(nm.group(1)) if nm else SUCCESS_CHUNKS

        with state.lock:
            state.counts[sid] = state.counts.get(sid, 0) + 1
            attempt = state.counts[sid]
            state.stream_flags.setdefault(sid, []).append(bool(body.get("stream")))
            status = 500 if behavior == "retry500" and attempt <= 2 else 200
            state.statuses.setdefault(sid, []).append(status)

        if status == 500:
            data = json.dumps({"error": "transient failure"}).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        def _first_chunk() -> None:
            ev = state.first_chunk.get(sid)
            if ev is not None:
                ev.set()

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")

        if behavior == "disconnect" and attempt == 1:
            # first attempt: one partial chunk, then drop the connection
            # mid-stream (declared body is never delivered)
            self.send_header("Content-Length", "1000000")
            self.end_headers()
            partial = json.dumps({"choices": [{"delta": {"content": f"PARTIAL{sid} "}}]})
            try:
                self.wfile.write(f"data: {partial}\n\n".encode())
                self.wfile.flush()
                _first_chunk()
                time.sleep(hold)
            finally:
                with contextlib.suppress(OSError):
                    self.connection.shutdown(2)
                self.connection.close()
            return

        if behavior in ("slow", "cancel"):
            # a long slow stream; the client aborts it mid-flight
            self.send_header("Content-Length", "1000000")
            self.end_headers()
            for i in range(300):
                chunk = json.dumps({"choices": [{"delta": {"content": f"c{sid}-{i} "}}]})
                try:
                    self.wfile.write(f"data: {chunk}\n\n".encode())
                    self.wfile.flush()
                    if i == 0:
                        _first_chunk()
                except OSError:
                    return
                time.sleep(0.2)
            return

        # success: a complete streaming response, written chunk by chunk
        chunks = [{"choices": [{"delta": {"content": f"r{sid}-{i} "}}]} for i in range(nchunks)]
        tail = (
            'data: {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}\n\n'
            + "data: [DONE]\n\n"
        )
        declared = sum(len(("data: " + json.dumps(c) + "\n\n").encode()) for c in chunks)
        declared += len(tail.encode())
        self.send_header("Content-Length", str(declared))
        self.end_headers()
        for i, c in enumerate(chunks):
            self.wfile.write(("data: " + json.dumps(c) + "\n\n").encode())
            self.wfile.flush()
            if i == 0:
                _first_chunk()
            time.sleep(CHUNK_INTERVAL)
        self.wfile.write(tail.encode())
        self.wfile.flush()

    def log_message(self, *a):
        pass


class TestConcurrentSubagents(unittest.TestCase):
    """Concurrent sub-agents on one session against a real server."""

    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.server.daemon_threads = True
        self.state = ServerState()
        self.server.state = self.state
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        host, port = self.server.server_address
        self.url = f"http://{host}:{port}/v1"
        self.template = Client(
            base_url=self.url,
            api_key="test",
            model="fake",
            timeout=10,
            retry_max=5,
            retry_base_delay=0.01,
            retry_max_delay=0.05,
        )
        self.session = AgentSession(
            project_dir="/tmp/fakeproj",
            client=self.template,
            model="fake",
            registry=default_registry(),
        )
        self.session.subagent_client = self.template
        self.results: dict[str, str] = {}

    def tearDown(self):
        self.session.close()
        self.template.close()
        self.server.shutdown()
        self.server.server_close()

    # -- helpers --------------------------------------------------------
    def run_agent(self, sid: str, behavior: str, hold: float = 0.05, n: int = SUCCESS_CHUNKS):
        """Launch one sub-agent in a background thread; returns the thread."""
        prompt = f"BEHAVIOR:{behavior} ID:{sid} HOLD:{hold} N:{n}"
        t = threading.Thread(
            target=lambda: self.results.__setitem__(
                sid, self.session.run_subagent("subagent", f"desc-{sid}", prompt)
            ),
            daemon=True,
        )
        t.start()
        return t

    @staticmethod
    def expected(sid: str, n: int = SUCCESS_CHUNKS) -> str:
        return "".join(f"r{sid}-{i} " for i in range(n))

    def assert_done(self, t: threading.Thread, timeout: float = 30) -> None:
        t.join(timeout=timeout)
        self.assertFalse(t.is_alive(), "sub-agent worker did not finish")

    # -- the review's race scenarios -------------------------------------
    def test_a_connection_drop_b_connection_remains_healthy(self):
        """A's connection drops mid-stream (and is retried on ITS OWN
        clone) while B is still streaming: B must complete on its FIRST
        attempt — the pre-fix ``_reset_http`` closed the shared pool and
        killed B's in-flight request."""
        self.state.first_chunk["A"] = threading.Event()
        self.state.first_chunk["B"] = threading.Event()
        tA = self.run_agent("A", "disconnect", hold=0.35)
        self.assertTrue(self.state.first_chunk["A"].wait(timeout=5))
        tB = self.run_agent("B", "success", n=16)  # still streaming while A drops
        self.assertTrue(self.state.first_chunk["B"].wait(timeout=5))
        self.assert_done(tA)
        self.assert_done(tB)
        # A: the dropped attempt was retried; the partial first-attempt
        # text must NOT appear in the final response (no duplication)
        self.assertGreaterEqual(self.state.counts["A"], 2)
        self.assertEqual(self.results["A"], self.expected("A"))
        self.assertNotIn("PARTIALA", self.results["A"])
        # B: untouched — exactly one request, full content
        self.assertEqual(self.state.counts["B"], 1)
        self.assertEqual(self.results["B"], self.expected("B", 16))

    def test_a_abort_b_continues(self):
        """One sub-agent's transport abort must not affect a sibling.

        session.cancel() is intentionally session-wide (Ctrl-C stops
        the whole run, so it aborts every active clone); this test pins
        the per-request isolation the clones provide: aborting A's
        clone (its pool + ``_aborted`` flag) leaves B's clone — and its
        in-flight request — completely untouched."""
        self.state.first_chunk["A"] = threading.Event()
        self.state.first_chunk["B"] = threading.Event()
        tA = self.run_agent("A", "slow")
        self.assertTrue(self.state.first_chunk["A"].wait(timeout=5))
        tB = self.run_agent("B", "success")
        self.assertTrue(self.state.first_chunk["B"].wait(timeout=5))
        with self.session._subagent_clients_lock:
            clones = list(self.session._active_subagent_clients)
        self.assertEqual(len(clones), 2)
        a_clone, b_clone = clones  # A registered first
        a_clone.abort()
        # A stops promptly (its blocked read was interrupted) with a
        # network error; B completes normally on its first attempt
        self.assert_done(tA, timeout=10)
        self.assertTrue(self.results["A"].startswith("Error: network error"))
        self.assert_done(tB)
        self.assertEqual(self.results["B"], self.expected("B"))
        self.assertEqual(self.state.counts["B"], 1)
        self.assertFalse(b_clone._aborted)

    def test_a_retry_b_completes(self):
        """A hits two HTTP 500s and retries (server sees 3 requests)
        while B streams: B completes on its FIRST attempt."""
        tA = self.run_agent("A", "retry500")
        tB = self.run_agent("B", "success")
        self.assert_done(tA)
        self.assert_done(tB)
        self.assertEqual(self.state.statuses["A"], [500, 500, 200])
        self.assertEqual(self.state.counts["A"], 3)
        self.assertEqual(self.results["A"], self.expected("A"))
        self.assertEqual(self.state.counts["B"], 1)
        self.assertEqual(self.results["B"], self.expected("B"))

    def test_a_stream_b_stream(self):
        """Both sub-agents stream concurrently and succeed: every
        request observed stream=true, one attempt each, exact content."""
        self.state.first_chunk["A"] = threading.Event()
        self.state.first_chunk["B"] = threading.Event()
        tA = self.run_agent("A", "success")
        tB = self.run_agent("B", "success")
        self.assertTrue(self.state.first_chunk["A"].wait(timeout=5))
        self.assertTrue(self.state.first_chunk["B"].wait(timeout=5))
        self.assert_done(tA)
        self.assert_done(tB)
        self.assertEqual(self.state.counts["A"], 1)
        self.assertEqual(self.state.counts["B"], 1)
        self.assertEqual(self.state.stream_flags["A"], [True])
        self.assertEqual(self.state.stream_flags["B"], [True])
        self.assertEqual(self.results["A"], self.expected("A"))
        self.assertEqual(self.results["B"], self.expected("B"))

    # -- the advanced scenario ------------------------------------------
    def test_50_concurrent_subagents_mixed_behaviors(self):
        """50 concurrent sub-agents with mixed behaviors — success,
        500-then-retry, disconnect-then-retry, cancel — must each end
        with exactly their own final response: no cross-contamination,
        no duplication (partial streams dropped on retry), no loss, no
        error pollution.  The cancelled ones stop promptly."""
        plan: list[tuple[int, str]] = []
        for sid in range(1, 51):
            if sid % 5 == 0:
                plan.append((sid, "retry500"))
            elif sid % 7 == 0:
                plan.append((sid, "disconnect"))
            elif sid % 11 == 0:
                plan.append((sid, "cancel"))
            else:
                plan.append((sid, "success"))
        self.assertEqual({b for _, b in plan}, {"success", "retry500", "disconnect", "cancel"})

        threads = {}
        for sid, behavior in plan:
            threads[str(sid)] = self.run_agent(str(sid), behavior)

        # the non-cancelled agents finish on their own
        for sid, behavior in plan:
            if behavior != "cancel":
                self.assert_done(threads[str(sid)], timeout=30)
        self.assertFalse(self.template._http.is_closed)

        # every finished agent: exact own content, no error pollution,
        # server-side attempt/status history matches the behavior
        for sid, behavior in plan:
            if behavior == "cancel":
                continue
            s = str(sid)
            self.assertEqual(self.results[s], self.expected(s), f"agent {s} polluted")
            self.assertNotIn("Error", self.results[s])
            self.assertNotIn("PARTIAL", self.results[s])
            counts = self.state.counts[s]
            if behavior == "success":
                self.assertEqual(counts, 1)
                self.assertEqual(self.state.statuses[s], [200])
            elif behavior == "retry500":
                self.assertEqual(counts, 3)
                self.assertEqual(self.state.statuses[s], [500, 500, 200])
            else:  # disconnect: first attempt killed, then success
                self.assertGreaterEqual(counts, 2)
                self.assertEqual(self.state.statuses[s], [200] * counts)
            self.assertTrue(all(self.state.stream_flags[s]))

        # no duplication / no loss: every expected token appears exactly
        # once across ALL results (and no foreign token leaked in)
        all_text = "".join(self.results[str(sid)] for sid, b in plan if b != "cancel")
        for sid, behavior in plan:
            if behavior == "cancel":
                continue
            for i in range(SUCCESS_CHUNKS):
                token = f"r{sid}-{i}"
                self.assertEqual(all_text.count(token), 1, f"{token} duplicated/lost")
        self.assertNotIn("PARTIAL", all_text)

        # now cancel: the blocked agents stop promptly with an error
        # string, their clones are released, the template pool survives
        self.session.cancel()
        for sid, behavior in plan:
            if behavior == "cancel":
                self.assert_done(threads[str(sid)], timeout=10)
                self.assertTrue(
                    self.results[str(sid)].startswith("Error:"),
                    f"cancelled agent {sid} did not stop cleanly: {self.results[str(sid)]!r}",
                )
        deadline = time.time() + 5
        while self.session._active_subagent_clients and time.time() < deadline:
            time.sleep(0.05)
        self.assertEqual(self.session._active_subagent_clients, [])
        self.assertFalse(self.template._http.is_closed)


if __name__ == "__main__":
    unittest.main()
