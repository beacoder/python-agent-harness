"""Client streaming tests against the in-process fake OpenAI server."""

import json
import os
import sys
import tempfile
import time
import unittest
from unittest import mock

import httpx

from python_agent_harness import config
from python_agent_harness.client import Client
from python_agent_harness.models import Message, ToolCall, ToolSpec, Usage

# `discover -s tests` puts the tests dir on sys.path, but a direct
# `-m unittest tests.test_client` invocation does not — make the
# sibling helper importable either way.
sys.path.insert(0, os.path.dirname(__file__))

import fake_openai_server  # noqa: E402  (state overrides for sync tests)
from fake_openai_server import serve  # noqa: E402


def make_client(
    retry_max: int | None = None,
    retry_base_delay: float | None = None,
    retry_max_delay: float | None = None,
) -> Client:
    srv = serve()
    host, port = srv.server_address
    c = Client(
        base_url=f"http://{host}:{port}/v1",
        api_key="test",
        model="fake",
        retry_max=retry_max,
        retry_base_delay=retry_base_delay,
        retry_max_delay=retry_max_delay,
    )
    c._server = srv  # keep the server alive for the test
    return c


class TestClientStreaming(unittest.TestCase):
    def test_reasoning_content_streamed_and_captured(self):
        """reasoning_content deltas stream normally (on_delta) AND are
        captured on the message so the TUI can collapse them later;
        tool-call fragments accumulate into complete ToolCalls."""
        c = make_client()
        deltas: list[str] = []
        tc_fragments: list[tuple[str, str, str]] = []
        msg, usage = c.chat(
            [Message(role="user", content="hi")],
            on_delta=deltas.append,
            on_tool_call=lambda name, tid, frag: tc_fragments.append((name, tid, frag)),
        )
        self.assertEqual("".join(deltas), "thinking hardHello world")
        self.assertEqual(msg.content, "thinking hardHello world")
        self.assertEqual(msg.reasoning, "thinking hard")
        self.assertIn("Hello world", msg.content)
        self.assertEqual(usage.input_tokens, 12)
        # the streamed Read tool-call chunk was reassembled
        self.assertEqual(len(msg.tool_calls), 1)
        self.assertEqual(msg.tool_calls[0].id, "call_1")
        self.assertEqual(msg.tool_calls[0].name, "Read")
        self.assertEqual(msg.tool_calls[0].arguments, '{"file_path": "/tmp/x.py"}')
        self.assertEqual(
            tc_fragments,
            [("Read", "call_1", '{"file_path": "/tmp/x.py"}')],
        )
        c.close()

    def test_streaming_default_sends_stream_true(self):
        """The default mode is streaming: the request body must carry
        stream=True plus stream_options requesting usage chunks."""
        fake_openai_server.reset_state()
        c = make_client()
        try:
            c.chat([Message(role="user", content="hi")])
        finally:
            c.close()
        body = fake_openai_server.REQUEST_BODIES[-1]
        self.assertIs(body["stream"], True)
        self.assertEqual(body["stream_options"], {"include_usage": True})


class TestClientNonStreaming(unittest.TestCase):
    def setUp(self):
        fake_openai_server.reset_state()

    def tearDown(self):
        fake_openai_server.reset_state()

    def test_sync_chat_returns_full_response(self):
        """stream=False performs a single POST and delivers the whole
        answer; on_delta fires once with the complete text."""
        c = make_client()
        deltas: list[str] = []
        try:
            msg, usage = c.chat(
                [Message(role="user", content="hi")],
                stream=False,
                on_delta=deltas.append,
            )
        finally:
            c.close()
        self.assertEqual(msg.content, "sync reply")
        self.assertIsNone(msg.reasoning)
        self.assertIsNone(msg.tool_calls)
        self.assertEqual(deltas, ["sync reply"])
        self.assertEqual(usage.input_tokens, 5)
        self.assertEqual(usage.output_tokens, 2)

    def test_sync_chat_sends_stream_false(self):
        """Non-streaming mode must send stream=False in the payload
        (and not ask for text/event-stream); stream_options must be
        absent since OpenAI-style backends reject it when not streaming."""
        c = make_client()
        try:
            c.chat([Message(role="user", content="hi")], stream=False)
        finally:
            c.close()
        body = fake_openai_server.REQUEST_BODIES[-1]
        self.assertIs(body["stream"], False)
        self.assertNotIn("stream_options", body)

    def test_sync_chat_tool_calls_and_reasoning(self):
        """stream=False parses content, reasoning_content, tool_calls
        and usage from the single response; callbacks fire once with the
        complete values (reasoning first, mirroring streaming order)."""
        fake_openai_server.NON_STREAM_RESPONSE = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "the answer",
                        "reasoning_content": "mulling",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "Read",
                                    "arguments": '{"file_path": "/tmp/a.py"}',
                                },
                            },
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {
                                    "name": "Grep",
                                    "arguments": '{"regex": "x", "path": "/tmp"}',
                                },
                            },
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 7, "completion_tokens": 9},
        }
        c = make_client()
        deltas: list[str] = []
        tc_calls: list[tuple[str, str, str]] = []
        try:
            msg, usage = c.chat(
                [Message(role="user", content="hi")],
                stream=False,
                on_delta=deltas.append,
                on_tool_call=lambda name, tid, frag: tc_calls.append((name, tid, frag)),
            )
        finally:
            c.close()
        self.assertEqual(msg.reasoning, "mulling")
        # content includes reasoning first, exactly like the streaming path
        self.assertEqual(msg.content, "mullingthe answer")
        self.assertEqual(deltas, ["mulling", "the answer"])
        self.assertEqual(len(msg.tool_calls), 2)
        self.assertEqual(msg.tool_calls[0].id, "call_1")
        self.assertEqual(msg.tool_calls[0].name, "Read")
        self.assertEqual(msg.tool_calls[0].arguments, '{"file_path": "/tmp/a.py"}')
        self.assertEqual(msg.tool_calls[1].name, "Grep")
        self.assertEqual(msg.tool_calls[1].arguments, '{"regex": "x", "path": "/tmp"}')
        self.assertEqual(
            tc_calls,
            [
                ("Read", "call_1", '{"file_path": "/tmp/a.py"}'),
                ("Grep", "call_2", '{"regex": "x", "path": "/tmp"}'),
            ],
        )
        self.assertEqual(usage.input_tokens, 7)
        self.assertEqual(usage.output_tokens, 9)

    def test_sync_chat_dict_arguments_normalized(self):
        """Some backends return tool-call arguments as an object rather
        than a JSON string; they must be normalized to a string."""
        fake_openai_server.NON_STREAM_RESPONSE = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_9",
                                "type": "function",
                                "function": {"name": "Bash", "arguments": {"command": "ls -la"}},
                            },
                        ],
                    }
                }
            ],
        }
        c = make_client()
        try:
            msg, _ = c.chat([Message(role="user", content="hi")], stream=False)
        finally:
            c.close()
        self.assertEqual(len(msg.tool_calls), 1)
        self.assertEqual(msg.tool_calls[0].name, "Bash")
        self.assertEqual(
            json.loads(msg.tool_calls[0].arguments),
            {"command": "ls -la"},
        )

    def test_sync_chat_list_content_normalized(self):
        """Some backends return content as a list of text parts
        (multimodal shape); they must be flattened to a plain string
        like Message.text() does — not crash the join."""
        fake_openai_server.NON_STREAM_RESPONSE = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "part one "},
                            {"type": "text", "text": "part two"},
                        ],
                    }
                }
            ],
        }
        c = make_client()
        try:
            msg, _ = c.chat([Message(role="user", content="hi")], stream=False)
        finally:
            c.close()
        self.assertEqual(msg.content, "part one part two")
        self.assertIsNone(msg.reasoning)

    def test_sync_chat_explicit_tool_call_index(self):
        """When a non-streaming response carries explicit index fields
        (mirroring the streaming shape), they must be honored."""
        fake_openai_server.NON_STREAM_RESPONSE = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "index": 1,
                                "id": "call_b",
                                "type": "function",
                                "function": {"name": "Grep", "arguments": "{}"},
                            },
                            {
                                "index": 0,
                                "id": "call_a",
                                "type": "function",
                                "function": {"name": "Read", "arguments": "{}"},
                            },
                        ],
                    }
                }
            ],
        }
        c = make_client()
        try:
            msg, _ = c.chat([Message(role="user", content="hi")], stream=False)
        finally:
            c.close()
        self.assertEqual([tc.id for tc in msg.tool_calls], ["call_a", "call_b"])
        self.assertEqual(msg.tool_calls[0].name, "Read")
        self.assertEqual(msg.tool_calls[1].name, "Grep")

    def test_sync_chat_non_string_reasoning_ignored(self):
        """Backends that report reasoning_content in a non-string shape
        (e.g. a list of parts) must not crash the parser; the reasoning
        is dropped instead of leaking into the content."""
        fake_openai_server.NON_STREAM_RESPONSE = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "answer",
                        "reasoning_content": [{"type": "text", "text": "mulling"}],
                    }
                }
            ],
        }
        c = make_client()
        try:
            msg, _ = c.chat([Message(role="user", content="hi")], stream=False)
        finally:
            c.close()
        self.assertEqual(msg.content, "answer")
        self.assertIsNone(msg.reasoning)

    def test_sync_chat_missing_choice_no_crash(self):
        """A non-streaming response with no choices at all yields an
        empty assistant message instead of crashing."""
        fake_openai_server.NON_STREAM_RESPONSE = {"choices": []}
        c = make_client()
        try:
            msg, _ = c.chat([Message(role="user", content="hi")], stream=False)
        finally:
            c.close()
        self.assertEqual(msg.content, "")
        self.assertIsNone(msg.tool_calls)

    def test_chat_sync_still_works(self):
        """chat_sync (compaction/titles/summary) delegates to the
        non-streaming path and keeps its behavior."""
        c = make_client()
        try:
            msg, usage = c.chat_sync([Message(role="user", content="hi")])
        finally:
            c.close()
        self.assertEqual(msg.content, "sync reply")
        self.assertEqual(usage.input_tokens, 5)
        self.assertEqual(usage.output_tokens, 2)
        self.assertIs(fake_openai_server.REQUEST_BODIES[-1]["stream"], False)

    def test_sync_error_raises_api_error(self):
        """Non-streaming HTTP errors surface as ApiError, like streaming.

        retry_max=1 disables retries so the error surfaces directly
        (no backoff sleep in the test).
        """
        from unittest import mock

        from python_agent_harness.client import ApiError

        c = make_client(retry_max=1)
        try:
            with mock.patch.object(c._http, "post") as post:
                post.return_value.status_code = 429
                post.return_value.text = "rate limited"
                with self.assertRaises(ApiError):
                    c.chat([Message(role="user", content="hi")], stream=False)
        finally:
            c.close()


class TestClientRetry(unittest.TestCase):
    """Transient API failures (429/5xx/connection) are retried with
    backoff; permanent errors are not; streamed output is never
    duplicated."""

    def setUp(self):
        fake_openai_server.reset_state()

    def tearDown(self):
        fake_openai_server.reset_state()

    def make_fast_client(self, retry_max: int = 3) -> Client:
        return make_client(retry_max=retry_max, retry_base_delay=0.01, retry_max_delay=0.05)

    def test_transient_errors_retried_then_success(self):
        """429 then 500 then success: the client retries with backoff
        and returns the final response unchanged."""
        fake_openai_server.STATUS_QUEUE = [429, 500]
        c = self.make_fast_client()
        try:
            msg, _ = c.chat([Message(role="user", content="hi")], stream=False)
        finally:
            c.close()
        self.assertEqual(msg.content, "sync reply")
        self.assertEqual(len(fake_openai_server.REQUEST_BODIES), 3)

    def test_permanent_error_not_retried(self):
        """A 4xx other than 429 is permanent: exactly one request."""
        from python_agent_harness.client import ApiError

        fake_openai_server.STATUS_QUEUE = [400]
        c = self.make_fast_client()
        try:
            with self.assertRaises(ApiError):
                c.chat([Message(role="user", content="hi")], stream=False)
        finally:
            c.close()
        self.assertEqual(len(fake_openai_server.REQUEST_BODIES), 1)

    def test_retry_budget_exhausted_raises(self):
        """Persistent 429s exhaust the attempt budget and the last
        transient error surfaces (as RetryableApiError)."""
        from python_agent_harness.client import RetryableApiError

        fake_openai_server.STATUS_QUEUE = [429, 429, 429, 429]
        c = self.make_fast_client()
        try:
            with self.assertRaises(RetryableApiError):
                c.chat([Message(role="user", content="hi")], stream=False)
        finally:
            c.close()
        self.assertEqual(len(fake_openai_server.REQUEST_BODIES), 3)

    def test_retry_after_header_honored(self):
        """A 429 carrying Retry-After is retried after that delay."""
        fake_openai_server.STATUS_QUEUE = [429]
        fake_openai_server.RETRY_AFTER_HEADER = "0.01"
        c = self.make_fast_client()
        try:
            msg, _ = c.chat([Message(role="user", content="hi")], stream=False)
        finally:
            c.close()
        self.assertEqual(msg.content, "sync reply")
        self.assertEqual(len(fake_openai_server.REQUEST_BODIES), 2)

    def test_streaming_retry_does_not_duplicate_deltas(self):
        """A stream that fails before any delta (429) is retried; the
        caller receives the streamed text exactly once."""
        fake_openai_server.STATUS_QUEUE = [429]
        c = self.make_fast_client()
        deltas: list[str] = []
        try:
            msg, _ = c.chat([Message(role="user", content="hi")], on_delta=deltas.append)
        finally:
            c.close()
        self.assertEqual("".join(deltas), "thinking hardHello world")
        self.assertEqual(msg.content, "thinking hardHello world")
        self.assertEqual(len(fake_openai_server.REQUEST_BODIES), 2)

    def test_cancel_check_aborts_backoff_sleep(self):
        """cancel_check=True ends the backoff sleep immediately; the
        pending transient error surfaces instead of a full wait."""
        from python_agent_harness.client import RetryableApiError

        fake_openai_server.STATUS_QUEUE = [429, 429]
        c = make_client(retry_max=3, retry_base_delay=60.0, retry_max_delay=60.0)
        try:
            with self.assertRaises(RetryableApiError):
                c.chat(
                    [Message(role="user", content="hi")],
                    stream=False,
                    cancel_check=lambda: True,
                )
        finally:
            c.close()
        self.assertEqual(len(fake_openai_server.REQUEST_BODIES), 1)


class FakeStreamResp:
    """Minimal stand-in for an httpx streaming response (used to exercise
    SSE parsing edge cases without a network round-trip)."""

    def __init__(self, lines, status_code=200, headers=None):
        self._lines = list(lines)
        self.status_code = status_code
        self.headers = headers or {}

    def read(self):
        return b"stream error body"

    def iter_lines(self):
        return iter(self._lines)


class FakeStreamCM:
    """Context manager yielding a FakeStreamResp, mimicking
    httpx.Client.stream()."""

    def __init__(self, resp):
        self.resp = resp

    def __enter__(self):
        return self.resp

    def __exit__(self, *exc):
        return False


def make_offline_client(**kwargs) -> Client:
    """Client that never talks to a real server (for unit-level paths)."""
    kwargs.setdefault("base_url", "http://127.0.0.1:1/v1")
    kwargs.setdefault("api_key", "test")
    kwargs.setdefault("model", "fake")
    return Client(**kwargs)


class TestClientHelpers(unittest.TestCase):
    """Unit tests for client plumbing: retry backoff, log paths, CA
    bundle resolution."""

    def test_retry_delay_honors_retry_after(self):
        from python_agent_harness.client import _retry_delay

        delay = _retry_delay(1, "1.5", 1.0, 30.0)
        self.assertGreaterEqual(delay, 1.5)
        self.assertLess(delay, 2.0)

    def test_retry_delay_invalid_retry_after_falls_back(self):
        """A non-numeric Retry-After must be ignored, not crash the
        backoff computation."""
        from python_agent_harness.client import _retry_delay

        delay = _retry_delay(1, "not-a-number", 1.0, 30.0)
        self.assertGreaterEqual(delay, 1.0)
        self.assertLess(delay, 1.31)

    def test_retry_delay_exponential_cap(self):
        from python_agent_harness.client import _retry_delay

        delay = _retry_delay(5, None, 1.0, 3.0)
        self.assertGreaterEqual(delay, 3.0)
        self.assertLess(delay, 3.91)

    def test_llm_log_path_uses_env_dir(self):
        """LLM_LOG_DIR wins: the log lands inside it and the directory
        is created on demand."""
        from python_agent_harness.client import _llm_log_path

        with tempfile.TemporaryDirectory(prefix="pah-log-") as d:
            env = os.environ.copy()
            env["LLM_LOG_DIR"] = d
            with mock.patch.dict(os.environ, env):
                p = _llm_log_path()
            self.assertTrue(str(p).startswith(d))
            self.assertTrue(os.path.isdir(d))
            self.assertRegex(p.name, r"^python-agent-harness-\d{8}-[0-9a-f]{8}\.json$")

    def test_llm_log_path_defaults_to_tmp(self):
        from python_agent_harness.client import _llm_log_path

        env = os.environ.copy()
        env.pop("LLM_LOG_DIR", None)
        with mock.patch.dict(os.environ, env):
            p = _llm_log_path()
        self.assertTrue(str(p).startswith("/tmp/"))
        self.assertRegex(p.name, r"^python-agent-harness-\d{8}-[0-9a-f]{8}\.json$")

    def test_resolve_ca_bundle_prefers_env(self):
        """An explicit SSL_CERT_FILE pointing at an existing file wins
        over the system bundle search."""
        from python_agent_harness.client import _resolve_ca_bundle

        with tempfile.NamedTemporaryFile(prefix="pah-ca-", suffix=".crt") as f:
            env = os.environ.copy()
            env["SSL_CERT_FILE"] = f.name
            with mock.patch.dict(os.environ, env):
                self.assertEqual(_resolve_ca_bundle(), f.name)

    def test_resolve_ca_bundle_falls_back_to_true(self):
        """When no env var and no system bundle exists, httpx's default
        verification (True) is used."""
        from python_agent_harness.client import _resolve_ca_bundle

        env = os.environ.copy()
        env.pop("SSL_CERT_FILE", None)
        with (
            mock.patch.dict(os.environ, env),
            mock.patch("python_agent_harness.client.os.path.isfile", return_value=False),
        ):
            self.assertIs(_resolve_ca_bundle(), True)


class TestClientLogging(unittest.TestCase):
    """_log_llm_interaction appends request bodies and assistant
    responses as pretty JSON and must never raise."""

    @staticmethod
    def _parse_logs(text):
        """Parse the two pretty-printed JSON documents (marker, body)."""
        dec = json.JSONDecoder()
        marker, idx = dec.raw_decode(text)
        body, _ = dec.raw_decode(text[idx:].lstrip())
        return marker, body

    def _logged(self, tmpdir, payload, resp, usage):
        from python_agent_harness.client import _log_llm_interaction

        log_file = os.path.join(tmpdir, "llm.json")
        _log_llm_interaction(log_file, payload, resp, usage)
        with open(log_file, encoding="utf-8") as f:
            return f.read()

    def test_writes_request_body_and_assistant_reply(self):
        with tempfile.TemporaryDirectory(prefix="pah-log-") as d:
            payload = {
                "model": "fake",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "Read"}}],
            }
            resp = Message(
                role="assistant",
                content="hello",
                tool_calls=[ToolCall(id="c1", name="Read", arguments='{"file_path": "/tmp/x"}')],
            )
            text = self._logged(d, payload, resp, Usage(input_tokens=5, output_tokens=2))
        marker, body = self._parse_logs(text)
        self.assertEqual(marker["python-agent-harness"], "request body")
        self.assertEqual(body["model"], "fake")
        self.assertEqual(len(body["messages"]), 2)
        reply = body["messages"][-1]
        self.assertEqual(reply["role"], "assistant")
        self.assertEqual(reply["content"], "hello")
        self.assertEqual(
            reply["tool_calls"][0]["function"]["arguments"],
            '{"file_path": "/tmp/x"}',
        )

    def test_dict_arguments_json_encoded(self):
        """Non-string tool-call arguments are JSON-encoded in the log."""
        with tempfile.TemporaryDirectory(prefix="pah-log-") as d:
            resp = Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(id="c2", name="Bash", arguments={"command": "ls"})],
            )
            text = self._logged(d, {"model": "m", "messages": []}, resp, Usage())
        _, body = self._parse_logs(text)
        self.assertIn(
            '"command": "ls"',
            body["messages"][-1]["tool_calls"][0]["function"]["arguments"],
        )

    def test_no_text_no_tool_calls_still_logged(self):
        with tempfile.TemporaryDirectory(prefix="pah-log-") as d:
            text = self._logged(
                d, {"model": "m", "messages": []}, Message(role="assistant"), Usage()
            )
        _, body = self._parse_logs(text)
        self.assertEqual(body["messages"][-1], {"role": "assistant"})

    def test_none_log_file_is_noop(self):
        from python_agent_harness.client import _log_llm_interaction

        _log_llm_interaction(
            None, {"model": "m", "messages": []}, Message(role="assistant"), Usage()
        )

    def test_write_errors_swallowed(self):
        """A failing log write (here: a directory as the log file) must
        never raise — logging is best effort."""
        from python_agent_harness.client import _log_llm_interaction

        with tempfile.TemporaryDirectory(prefix="pah-log-") as d:
            _log_llm_interaction(
                d, {"model": "m", "messages": []}, Message(role="assistant"), Usage()
            )


class TestClientAbort(unittest.TestCase):
    """abort() swaps in a fresh transport and tolerates failures in the
    socket-waking and close steps."""

    def test_abort_swaps_client_and_closes_old(self):
        c = make_offline_client()
        old = c._http
        with (
            mock.patch("python_agent_harness.client._abort_inflight_sockets") as ab,
            mock.patch.object(old, "close") as cl,
        ):
            c.abort()
        ab.assert_called_once_with(old)
        cl.assert_called_once_with()
        self.assertIsNot(c._http, old)
        c.close()

    def test_abort_tolerates_socket_wake_failure(self):
        c = make_offline_client()
        old = c._http
        with mock.patch(
            "python_agent_harness.client._abort_inflight_sockets",
            side_effect=RuntimeError("boom"),
        ):
            c.abort()  # must not raise
        self.assertIsNot(c._http, old)
        c.close()

    def test_abort_tolerates_close_failure(self):
        c = make_offline_client()
        old = c._http
        with (
            mock.patch("python_agent_harness.client._abort_inflight_sockets"),
            mock.patch.object(old, "close", side_effect=RuntimeError("boom")),
        ):
            c.abort()  # must not raise
        c.close()


class TestAbortInflightSockets(unittest.TestCase):
    """_abort_inflight_sockets wakes live pool sockets via
    shutdown(SHUT_RDWR) and tolerates broken internals."""

    def _fake_client_with_stream(self, stream):
        conn = mock.Mock()
        conn._connection._network_stream = stream
        client = mock.Mock()
        client._transport._pool._connections = [conn]
        return client

    def test_shuts_down_live_socket(self):
        import socket as _socket

        from python_agent_harness.client import _abort_inflight_sockets

        sock = mock.Mock()
        stream = mock.Mock()
        stream.get_extra_info.return_value = sock
        _abort_inflight_sockets(self._fake_client_with_stream(stream))
        sock.shutdown.assert_called_once_with(_socket.SHUT_RDWR)

    def test_empty_pool_is_noop(self):
        from python_agent_harness.client import _abort_inflight_sockets

        client = mock.Mock()
        client._transport._pool._connections = []
        _abort_inflight_sockets(client)  # must not raise

    def test_get_extra_info_failure_ignored(self):
        from python_agent_harness.client import _abort_inflight_sockets

        stream = mock.Mock()
        stream.get_extra_info.side_effect = RuntimeError("no socket")
        _abort_inflight_sockets(self._fake_client_with_stream(stream))  # no raise

    def test_shutdown_oserror_ignored(self):
        from python_agent_harness.client import _abort_inflight_sockets

        sock = mock.Mock()
        sock.shutdown.side_effect = OSError("closed")
        stream = mock.Mock()
        stream.get_extra_info.return_value = sock
        _abort_inflight_sockets(self._fake_client_with_stream(stream))  # no raise


class TestClientPayloadAndHeaders(unittest.TestCase):
    """Request payload/header construction: tools, options, system
    prompt, stream Accept header, api-key authorization."""

    def setUp(self):
        fake_openai_server.reset_state()

    def tearDown(self):
        fake_openai_server.reset_state()

    def test_payload_carries_tools_and_options(self):
        c = make_client()
        try:
            c.chat(
                [Message(role="user", content="hi")],
                tools=[ToolSpec(name="Read", description="read", parameters={"type": "object"})],
                temperature=0.7,
                max_tokens=123,
                reasoning_effort="high",
                stream=False,
            )
        finally:
            c.close()
        body = fake_openai_server.REQUEST_BODIES[-1]
        self.assertEqual(body["tools"][0]["function"]["name"], "Read")
        self.assertEqual(body["temperature"], 0.7)
        self.assertEqual(body["max_tokens"], 123)
        self.assertEqual(body["reasoning_effort"], "high")

    def test_payload_system_prepended(self):
        c = make_client()
        try:
            c.chat([Message(role="user", content="hi")], system="be brief", stream=False)
        finally:
            c.close()
        msgs = fake_openai_server.REQUEST_BODIES[-1]["messages"]
        self.assertEqual(msgs[0], {"role": "system", "content": "be brief"})
        self.assertEqual(msgs[1]["role"], "user")

    def test_headers_stream_accept_and_api_key(self):
        c = make_client()
        try:
            self.assertEqual(c._headers(stream=True)["Accept"], "text/event-stream")
            self.assertEqual(c._headers(stream=False)["Accept"], "application/json")
            self.assertEqual(c._headers()["Authorization"], "Bearer test")
        finally:
            c.close()

    def test_no_api_key_omits_authorization(self):
        env = os.environ.copy()
        env.pop("OPENAI_API_KEY", None)
        env.pop("DEEPSEEK_API_KEY", None)
        with mock.patch.dict(os.environ, env):
            c = Client(base_url="http://127.0.0.1:1/v1", api_key=None, model="fake")
            try:
                self.assertNotIn("Authorization", c._headers())
            finally:
                c.close()


class TestClientStreamingEdgeCases(unittest.TestCase):
    """Streaming parsing edge cases exercised with a stubbed stream:
    permanent 4xx, empty/garbage/no-choice chunks."""

    def test_stream_permanent_error_raises_api_error(self):
        """A 4xx other than 429 is permanent even in streaming mode:
        exactly one request, no retry."""
        from python_agent_harness.client import ApiError

        c = make_client(retry_max=3)
        try:
            resp = FakeStreamResp([], status_code=400)
            with (
                mock.patch.object(c._http, "stream", return_value=FakeStreamCM(resp)),
                self.assertRaises(ApiError),
            ):
                c.chat([Message(role="user", content="hi")])
        finally:
            c.close()

    def test_stream_skips_empty_garbage_and_choice_free_chunks(self):
        """Empty SSE payloads, non-JSON data payloads and chunks without
        choices are skipped without failing the stream."""
        c = make_client()
        try:
            lines = [
                "data:",  # empty SSE payload
                "data: not json at all",
                'data: {"choices": []}',
                'data: {"choices": [{"delta": {"content": "ok"}}]}',
                "data: [DONE]",
            ]
            resp = FakeStreamResp(lines)
            with mock.patch.object(c._http, "stream", return_value=FakeStreamCM(resp)):
                msg, usage = c.chat([Message(role="user", content="hi")])
            self.assertEqual(msg.content, "ok")
            self.assertEqual(usage.input_tokens, 0)
        finally:
            c.close()


class TestClientNetworkErrors(unittest.TestCase):
    """Connection-level failures (httpx.HTTPError) are retried until the
    budget is exhausted, then surface as ApiError."""

    def test_connect_error_retried_then_raises(self):
        from python_agent_harness.client import ApiError

        c = make_client(retry_max=2, retry_base_delay=0.01, retry_max_delay=0.05)
        try:
            with (
                mock.patch.object(c._http, "stream", side_effect=httpx.ConnectError("refused")),
                self.assertRaises(ApiError) as ctx,
            ):
                c.chat([Message(role="user", content="hi")], cancel_check=lambda: True)
            self.assertIn("network error", str(ctx.exception))
        finally:
            c.close()

    def test_connect_error_non_stream_retried_then_raises(self):
        from python_agent_harness.client import ApiError

        c = make_client(retry_max=2, retry_base_delay=0.01, retry_max_delay=0.05)
        try:
            # patch the class so the fresh client created by
            # _reset_http between attempts is covered too
            with (
                mock.patch.object(httpx.Client, "post", side_effect=httpx.ConnectError("refused")),
                self.assertRaises(ApiError),
            ):
                c.chat([Message(role="user", content="hi")], stream=False)
        finally:
            c.close()

    def test_budget_exhausted_immediately_raises(self):
        from python_agent_harness.client import ApiError

        c = make_client(retry_max=1)
        try:
            with (
                mock.patch.object(c._http, "stream", side_effect=httpx.ReadError("dropped")),
                self.assertRaises(ApiError),
            ):
                c.chat([Message(role="user", content="hi")])
        finally:
            c.close()

    def test_error_after_delta_retried_on_fresh_client(self):
        """A stream dying mid-body after deltas is retried on a fresh
        client: the partial stream is discarded, the retried response
        is returned, and nothing is duplicated in the stored message."""
        c = make_client(retry_max=2, retry_base_delay=0.01, retry_max_delay=0.01)
        old = c._http
        try:
            state = {"n": 0}

            def flaky(self_, *a, **kw):
                state["n"] += 1
                if state["n"] == 1:
                    return DieAfterDelta()
                resp = FakeStreamResp(
                    [
                        'data: {"choices": [{"delta": {"content": "full"}}]}',
                        "data: [DONE]",
                    ]
                )
                return FakeStreamCM(resp)

            # patch the class so the fresh client created by
            # _reset_http is covered too
            with mock.patch.object(httpx.Client, "stream", flaky):
                retries = []
                deltas = []
                msg, _ = c.chat(
                    [Message(role="user", content="hi")],
                    on_delta=deltas.append,
                    on_retry=lambda: retries.append(1),
                )
            self.assertEqual(msg.content, "full")
            self.assertEqual(state["n"], 2)
            self.assertEqual(retries, [1])
            # the retry ran on a fresh (reset) client
            self.assertIsNot(c._http, old)
            # the caller saw the partial delta, but the stored message
            # only carries the retried response
            self.assertEqual(deltas, ["partial", "full"])
        finally:
            c.close()

    def test_error_after_delta_retries_exhausted_then_raises(self):
        """A stream dying mid-body after deltas retries (reset client
        each time); when the attempt budget is exhausted it raises
        ApiError — with the client left fresh for the next call."""
        from python_agent_harness.client import ApiError

        c = make_client(retry_max=2, retry_base_delay=0.01, retry_max_delay=0.01)
        old = c._http
        try:
            with mock.patch.object(httpx.Client, "stream", return_value=DieAfterDelta()):
                retries = []
                with self.assertRaises(ApiError):
                    c.chat(
                        [Message(role="user", content="hi")],
                        on_retry=lambda: retries.append(1),
                    )
            self.assertEqual(retries, [1])  # one retry after the first drop
            self.assertIsNot(c._http, old)  # pool was reset
            self.assertFalse(c._http.is_closed)
        finally:
            c.close()


class DieAfterDelta:
    """Streaming response that yields one delta then dies with a
    connection error (mimics a connection reset mid-body)."""

    def __init__(self):
        self._sent = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @property
    def status_code(self):
        return 200

    @property
    def headers(self):
        return {}

    def read(self):
        return b""

    def iter_lines(self):
        yield 'data: {"choices": [{"delta": {"content": "partial"}}]}'
        raise httpx.ReadError("connection reset")


class TestClientResetHttp(unittest.TestCase):
    """_reset_http replaces the httpx client so a poisoned connection
    pool does not doom subsequent requests."""

    def test_reset_http_replaces_client(self):
        c = make_offline_client()
        old = c._http
        c._reset_http()
        self.assertIsNot(c._http, old)
        self.assertFalse(c._http.is_closed)
        c.close()

    def test_reset_http_closes_old_client(self):
        c = make_offline_client()
        old = c._http
        with mock.patch.object(old, "close") as cl:
            c._reset_http()
        cl.assert_called_once_with()
        c.close()

    def test_reset_http_tolerates_close_failure(self):
        c = make_offline_client()
        old = c._http
        with mock.patch.object(old, "close", side_effect=RuntimeError("boom")):
            c._reset_http()  # must not raise
        self.assertIsNot(c._http, old)
        c.close()

    def test_connection_error_resets_pool_for_next_request(self):
        """After retries exhaust on a connection error, the next chat()
        call uses a fresh httpx client (not the poisoned one)."""
        from python_agent_harness.client import ApiError

        c = make_offline_client(retry_max=2, retry_base_delay=0.01, retry_max_delay=0.01)
        # Simulate persistent connection failures
        with (
            mock.patch.object(c._http, "stream", side_effect=httpx.ConnectError("refused")),
            self.assertRaises(ApiError),
        ):
            c.chat([Message(role="user", content="hi")])
        # After the error, _http must be a fresh (non-poisoned) client
        # that was NOT the one we patched
        self.assertFalse(c._http.is_closed)
        # Verify it's a different object (reset happened)
        # The patched mock is on the OLD client; the new one is real
        self.assertNotIsInstance(c._http.stream, mock.Mock)
        c.close()

    def test_cancel_during_backoff_also_resets_pool(self):
        """If cancel_check fires during retry backoff, the pool is
        still reset so the next run starts clean."""
        from python_agent_harness.client import ApiError

        c = make_offline_client(retry_max=3, retry_base_delay=60.0, retry_max_delay=60.0)
        old_http = c._http
        with (
            mock.patch.object(c._http, "stream", side_effect=httpx.ConnectError("refused")),
            self.assertRaises(ApiError),
        ):
            c.chat(
                [Message(role="user", content="hi")],
                cancel_check=lambda: True,
            )
        self.assertIsNot(c._http, old_http)
        self.assertFalse(c._http.is_closed)
        c.close()

    def test_bare_oserror_is_retried_and_resets_pool(self):
        """A raw OSError (e.g. ConnectionResetError / ssl.SSLError) that
        leaks past httpx must be treated like a connection error: it is
        retried on a fresh client and surfaces as ApiError.  Without
        this, the poisoned connection would stay in the pool and doom
        every following request in the session."""
        from python_agent_harness.client import ApiError

        c = make_offline_client(retry_max=3, retry_base_delay=0.01, retry_max_delay=0.01)
        old_http = c._http
        attempts = {"n": 0}

        def boom(*_a, **_k):
            attempts["n"] += 1
            raise ConnectionResetError(104, "Connection reset by peer")

        with (
            mock.patch.object(c, "_stream_response", side_effect=boom),
            self.assertRaises(ApiError),
        ):
            c.chat([Message(role="user", content="hi")])

        # retried up to the budget (not a single-shot failure)
        self.assertEqual(attempts["n"], 3)
        # pool reset: the poisoned client was swapped out for a fresh one
        self.assertIsNot(c._http, old_http)
        self.assertFalse(c._http.is_closed)
        c.close()

    def test_unexpected_error_resets_pool_but_is_not_retried(self):
        """A non-connection error (e.g. a bug, or a permanent ApiError)
        must fail fast — a single attempt, no retry — yet still reset
        the pool so the next request never reuses a poisoned connection."""
        from python_agent_harness.client import ApiError

        c = make_offline_client(retry_max=3, retry_base_delay=0.01, retry_max_delay=0.01)
        old_http = c._http
        attempts = {"n": 0}

        def boom(*_a, **_k):
            attempts["n"] += 1
            raise ApiError("API error 403: forbidden")

        with (
            mock.patch.object(c, "_stream_response", side_effect=boom),
            self.assertRaises(ApiError) as ctx,
        ):
            c.chat([Message(role="user", content="hi")])

        # exactly one attempt: permanent errors are NOT retried
        self.assertEqual(attempts["n"], 1)
        # original error propagates unchanged (not re-wrapped as network)
        self.assertIn("403", str(ctx.exception))
        # pool still reset so the next request starts clean
        self.assertIsNot(c._http, old_http)
        self.assertFalse(c._http.is_closed)
        c.close()


class TestClientCallbackIsolation(unittest.TestCase):
    """A presentational callback (on_delta / on_tool_call) failure must
    never be mistaken for a transport error and trigger a retry — even
    when it raises an OSError such as BrokenPipeError on a closed
    terminal."""

    def test_on_delta_oserror_does_not_trigger_retry(self):
        c = make_offline_client(retry_max=3, retry_base_delay=0.01, retry_max_delay=0.01)
        calls = {"n": 0}

        def fake_stream(payload, on_delta, on_tool_call, usage):
            on_delta("hello")  # invokes the wrapped user callback
            return (["hello"], [], {})

        def bad_on_delta(_chunk):
            calls["n"] += 1
            raise BrokenPipeError(32, "Broken pipe")

        with mock.patch.object(c, "_stream_response", side_effect=fake_stream):
            msg, _ = c.chat(
                [Message(role="user", content="hi")],
                on_delta=bad_on_delta,
                stream=True,
            )
        # request succeeded despite the UI callback blowing up, and it
        # ran exactly once (no spurious network retry)
        self.assertEqual(msg.text(), "hello")
        self.assertEqual(calls["n"], 1)
        c.close()

    def test_on_tool_call_exception_does_not_trigger_retry(self):
        c = make_offline_client(retry_max=3, retry_base_delay=0.01, retry_max_delay=0.01)
        calls = {"n": 0}

        def fake_stream(payload, on_delta, on_tool_call, usage):
            on_tool_call("Read", "call_1", '{"path":"x"}')
            return ([], [], {0: {"id": "call_1", "name": "Read", "arguments": '{"path":"x"}'}})

        def bad_on_tool_call(_n, _i, _f):
            calls["n"] += 1
            raise RuntimeError("render boom")

        with mock.patch.object(c, "_stream_response", side_effect=fake_stream):
            msg, _ = c.chat(
                [Message(role="user", content="hi")],
                on_tool_call=bad_on_tool_call,
                stream=True,
            )
        self.assertTrue(msg.tool_calls)
        self.assertEqual(calls["n"], 1)
        c.close()


class TestRetryAfterPartialStream(unittest.TestCase):
    """A transient status (429/5xx) arriving AFTER a partial stream was
    dropped (and cleared) must still be retried — emission is tracked
    per-attempt, not for the whole request."""

    def test_transient_status_after_dropped_partial_is_retried(self):
        from python_agent_harness.client import RetryableApiError

        c = make_offline_client(retry_max=3, retry_base_delay=0.01, retry_max_delay=0.01)
        calls = {"n": 0}
        deltas: list[str] = []
        retries = {"n": 0}

        def fake_stream(payload, on_delta, on_tool_call, usage):
            n = calls["n"]
            calls["n"] += 1
            if n == 0:
                on_delta("part")  # partial delivered to the caller
                raise httpx.ReadError("stream dropped mid-body")
            if n == 1:
                # transient status on the retry, AFTER a partial was
                # already streamed on attempt 0
                raise RetryableApiError("API error 429", None)
            return (["full answer"], [], {})

        with mock.patch.object(c, "_stream_response", side_effect=fake_stream):
            msg, _ = c.chat(
                [Message(role="user", content="hi")],
                on_delta=deltas.append,
                on_retry=lambda: retries.__setitem__("n", retries["n"] + 1),
                stream=True,
            )

        # all three attempts ran: the 429 after the dropped partial was
        # NOT treated as terminal (the pre-fix bug gave up here)
        self.assertEqual(calls["n"], 3)
        # final message is the last attempt's content, no duplication
        self.assertEqual(msg.text(), "full answer")
        # the dropped partial was cleared via on_retry at least once
        self.assertGreaterEqual(retries["n"], 1)
        c.close()


class TestAuthRefreshOn401(unittest.TestCase):
    """When a 401 is received, the client re-reads the API key from
    config/env.  If a new key is found, the request is retried once
    with the updated credentials.  If the key hasn't changed, the
    error propagates immediately."""

    def test_401_with_refreshed_key_retries_and_succeeds(self):
        """A 401 triggers key re-read; if the key changed, the request
        is retried with the new key and can succeed."""
        from python_agent_harness.client import AuthExpiredError

        c = make_offline_client(retry_max=3, retry_base_delay=0.01, retry_max_delay=0.01)
        calls = {"n": 0}

        def fake_stream(payload, on_delta, on_tool_call, usage):
            n = calls["n"]
            calls["n"] += 1
            if n == 0:
                raise AuthExpiredError("API error 401: Unauthorized")
            # second attempt succeeds with the refreshed key
            return (["ok"], [], {})

        with (
            mock.patch.object(c, "_stream_response", side_effect=fake_stream),
            mock.patch.object(c, "_refresh_api_key", return_value=True),
        ):
            msg, _ = c.chat([Message(role="user", content="hi")])

        # retried once after refresh
        self.assertEqual(calls["n"], 2)
        self.assertEqual(msg.text(), "ok")
        c.close()

    def test_401_with_unchanged_key_fails_immediately(self):
        """A 401 where the key hasn't changed on disk propagates as a
        permanent ApiError (no retry)."""
        from python_agent_harness.client import ApiError, AuthExpiredError

        c = make_offline_client(retry_max=3, retry_base_delay=0.01, retry_max_delay=0.01)
        calls = {"n": 0}

        def fake_stream(payload, on_delta, on_tool_call, usage):
            calls["n"] += 1
            raise AuthExpiredError("API error 401: Unauthorized")

        with (
            mock.patch.object(c, "_stream_response", side_effect=fake_stream),
            mock.patch.object(c, "_refresh_api_key", return_value=False),
            self.assertRaises(ApiError) as ctx,
        ):
            c.chat([Message(role="user", content="hi")])

        # exactly one attempt: key didn't change, no retry
        self.assertEqual(calls["n"], 1)
        self.assertIn("401", str(ctx.exception))
        c.close()

    def test_401_only_retries_once_even_if_key_keeps_changing(self):
        """Even if _refresh_api_key returns True repeatedly (a bug or
        race), the auth refresh retry is capped at one attempt to
        prevent infinite loops."""
        from python_agent_harness.client import ApiError, AuthExpiredError

        c = make_offline_client(retry_max=3, retry_base_delay=0.01, retry_max_delay=0.01)
        calls = {"n": 0}

        def fake_stream(payload, on_delta, on_tool_call, usage):
            calls["n"] += 1
            raise AuthExpiredError("API error 401: Unauthorized")

        with (
            mock.patch.object(c, "_stream_response", side_effect=fake_stream),
            mock.patch.object(c, "_refresh_api_key", return_value=True),
            self.assertRaises(ApiError) as ctx,
        ):
            c.chat([Message(role="user", content="hi")])

        # first attempt + one auth-refresh retry = 2 total
        self.assertEqual(calls["n"], 2)
        self.assertIn("401", str(ctx.exception))
        c.close()

    def test_401_resets_http_pool(self):
        """A 401 must reset the connection pool (same as other errors)
        so subsequent requests start clean."""
        from python_agent_harness.client import ApiError, AuthExpiredError

        c = make_offline_client(retry_max=3, retry_base_delay=0.01, retry_max_delay=0.01)
        old_http = c._http

        def fake_stream(payload, on_delta, on_tool_call, usage):
            raise AuthExpiredError("API error 401: Unauthorized")

        with (
            mock.patch.object(c, "_stream_response", side_effect=fake_stream),
            mock.patch.object(c, "_refresh_api_key", return_value=False),
            self.assertRaises(ApiError),
        ):
            c.chat([Message(role="user", content="hi")])

        # pool was swapped out
        self.assertIsNot(c._http, old_http)
        self.assertFalse(c._http.is_closed)
        c.close()

    def test_refresh_api_key_polls_until_key_changes(self):
        """_refresh_api_key polls the config file repeatedly until the
        key changes, then returns True."""
        import json
        import tempfile

        cfg_path = tempfile.mktemp(suffix=".json")
        # initially same key
        with open(cfg_path, "w") as f:
            json.dump({"llm": {"api_key": "old-token"}}, f)

        c = make_offline_client(config_path=cfg_path)
        c.api_key = "old-token"

        call_count = {"n": 0}
        original_load = config.load_llm_config

        def patched_load(path=None):
            call_count["n"] += 1
            if call_count["n"] >= 3:
                # simulate external script updating the file
                with open(cfg_path, "w") as f:
                    json.dump({"llm": {"api_key": "refreshed-token"}}, f)
            return original_load(path)

        try:
            with mock.patch.object(config, "load_llm_config", side_effect=patched_load):
                result = c._refresh_api_key(timeout=5.0, poll_interval=0.05)
            self.assertTrue(result)
            self.assertEqual(c.api_key, "refreshed-token")
            self.assertGreaterEqual(call_count["n"], 3)
        finally:
            os.unlink(cfg_path)
            c.close()

    def test_refresh_api_key_times_out_when_key_unchanged(self):
        """_refresh_api_key returns False after timeout if the key
        never changes."""
        import json
        import tempfile

        cfg = {"llm": {"api_key": "same-token"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(cfg, f)
            cfg_path = f.name

        try:
            c = make_offline_client(config_path=cfg_path)
            c.api_key = "same-token"
            start = time.monotonic()
            result = c._refresh_api_key(timeout=0.2, poll_interval=0.05)
            elapsed = time.monotonic() - start
            self.assertFalse(result)
            # actually waited close to the timeout
            self.assertGreaterEqual(elapsed, 0.15)
        finally:
            os.unlink(cfg_path)
            c.close()

    def test_refresh_api_key_aborts_on_cancel(self):
        """_refresh_api_key respects cancel_check and returns False
        early without waiting the full timeout."""
        import json
        import tempfile

        cfg = {"llm": {"api_key": "same-token"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(cfg, f)
            cfg_path = f.name

        try:
            c = make_offline_client(config_path=cfg_path)
            c.api_key = "same-token"
            start = time.monotonic()
            result = c._refresh_api_key(
                cancel_check=lambda: True,  # immediate cancel
                timeout=10.0,
                poll_interval=0.05,
            )
            elapsed = time.monotonic() - start
            self.assertFalse(result)
            # aborted quickly, not after 10s
            self.assertLess(elapsed, 1.0)
        finally:
            os.unlink(cfg_path)
            c.close()

    def test_refresh_api_key_falls_back_to_env(self):
        """When no config file exists, _refresh_api_key reads from
        environment variables."""
        c = make_offline_client(config_path="/nonexistent/config.json")
        c.api_key = "old-key"

        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "env-refreshed-key"}):
            result = c._refresh_api_key(timeout=0.1, poll_interval=0.05)
        self.assertTrue(result)
        self.assertEqual(c.api_key, "env-refreshed-key")
        c.close()

    def test_401_on_sync_request_also_triggers_refresh(self):
        """Non-streaming requests (chat_sync, used for titles) also
        benefit from the auth refresh mechanism."""
        from python_agent_harness.client import AuthExpiredError

        c = make_offline_client(retry_max=3, retry_base_delay=0.01, retry_max_delay=0.01)
        calls = {"n": 0}

        def fake_sync(payload, on_delta, on_tool_call, usage):
            n = calls["n"]
            calls["n"] += 1
            if n == 0:
                raise AuthExpiredError("API error 401: Unauthorized")
            return (["title generated"], [], {})

        with (
            mock.patch.object(c, "_sync_response", side_effect=fake_sync),
            mock.patch.object(c, "_refresh_api_key", return_value=True),
        ):
            msg, _ = c.chat_sync([Message(role="user", content="hi")])

        self.assertEqual(calls["n"], 2)
        self.assertEqual(msg.text(), "title generated")
        c.close()


if __name__ == "__main__":
    unittest.main()
