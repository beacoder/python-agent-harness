"""Client streaming tests against the in-process fake OpenAI server."""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import httpx

from python_agent_harness.client import Client
from python_agent_harness.models import Message, ToolCall, ToolSpec, Usage

# `discover -s tests` puts the tests dir on sys.path, but a direct
# `-m unittest tests.test_client` invocation does not — make the
# sibling helper importable either way.
sys.path.insert(0, os.path.dirname(__file__))

from fake_openai_server import serve  # noqa: E402
import fake_openai_server  # noqa: E402  (state overrides for sync tests)


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
        self.assertEqual(
            msg.tool_calls[0].arguments, '{"file_path": "/tmp/x.py"}'
        )
        self.assertEqual(
            tc_fragments,
            [("Read", "call_1", '{"file_path": "/tmp/x.py"}')],
        )
        c.close()

    def test_streaming_default_sends_stream_true(self):
        """The default mode is streaming: the request body must carry
        stream=True unless the caller opts out."""
        fake_openai_server.reset_state()
        c = make_client()
        try:
            c.chat([Message(role="user", content="hi")])
        finally:
            c.close()
        self.assertIs(fake_openai_server.REQUEST_BODIES[-1]["stream"], True)


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
        (and not ask for text/event-stream)."""
        c = make_client()
        try:
            c.chat([Message(role="user", content="hi")], stream=False)
        finally:
            c.close()
        self.assertIs(fake_openai_server.REQUEST_BODIES[-1]["stream"], False)

    def test_sync_chat_tool_calls_and_reasoning(self):
        """stream=False parses content, reasoning_content, tool_calls
        and usage from the single response; callbacks fire once with the
        complete values (reasoning first, mirroring streaming order)."""
        fake_openai_server.NON_STREAM_RESPONSE = {
            "choices": [{"message": {
                "role": "assistant",
                "content": "the answer",
                "reasoning_content": "mulling",
                "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "Read",
                                  "arguments": '{"file_path": "/tmp/a.py"}'}},
                    {"id": "call_2", "type": "function",
                     "function": {"name": "Grep",
                                  "arguments": '{"regex": "x", "path": "/tmp"}'}},
                ],
            }}],
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
            "choices": [{"message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_9", "type": "function",
                     "function": {"name": "Bash",
                                  "arguments": {"command": "ls -la"}}},
                ],
            }}],
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
            "choices": [{"message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "part one "},
                    {"type": "text", "text": "part two"},
                ],
            }}],
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
            "choices": [{"message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"index": 1, "id": "call_b", "type": "function",
                     "function": {"name": "Grep", "arguments": "{}"}},
                    {"index": 0, "id": "call_a", "type": "function",
                     "function": {"name": "Read", "arguments": "{}"}},
                ],
            }}],
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
            "choices": [{"message": {
                "role": "assistant",
                "content": "answer",
                "reasoning_content": [{"type": "text", "text": "mulling"}],
            }}],
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
        from python_agent_harness.client import ApiError
        from unittest import mock

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
        return make_client(
            retry_max=retry_max, retry_base_delay=0.01, retry_max_delay=0.05
        )

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
            msg, _ = c.chat(
                [Message(role="user", content="hi")], on_delta=deltas.append
            )
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
            self.assertRegex(
                p.name, r"^python-agent-harness-\d{8}-[0-9a-f]{8}\.json$"
            )

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
        with mock.patch.dict(os.environ, env):
            with mock.patch(
                "python_agent_harness.client.os.path.isfile", return_value=False
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
        with mock.patch("python_agent_harness.client._abort_inflight_sockets") as ab:
            with mock.patch.object(old, "close") as cl:
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
        with mock.patch("python_agent_harness.client._abort_inflight_sockets"):
            with mock.patch.object(old, "close", side_effect=RuntimeError("boom")):
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
            with mock.patch.object(c._http, "stream", return_value=FakeStreamCM(resp)):
                with self.assertRaises(ApiError):
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
            with mock.patch.object(
                c._http, "stream", side_effect=httpx.ConnectError("refused")
            ):
                with self.assertRaises(ApiError) as ctx:
                    c.chat([Message(role="user", content="hi")], cancel_check=lambda: True)
            self.assertIn("network error", str(ctx.exception))
        finally:
            c.close()

    def test_connect_error_non_stream_retried_then_raises(self):
        from python_agent_harness.client import ApiError

        c = make_client(retry_max=2, retry_base_delay=0.01, retry_max_delay=0.05)
        try:
            with mock.patch.object(
                c._http, "post", side_effect=httpx.ConnectError("refused")
            ):
                with self.assertRaises(ApiError):
                    c.chat([Message(role="user", content="hi")], stream=False)
        finally:
            c.close()

    def test_budget_exhausted_immediately_raises(self):
        from python_agent_harness.client import ApiError

        c = make_client(retry_max=1)
        try:
            with mock.patch.object(
                c._http, "stream", side_effect=httpx.ReadError("dropped")
            ):
                with self.assertRaises(ApiError):
                    c.chat([Message(role="user", content="hi")])
        finally:
            c.close()

    def test_error_after_delta_not_retried(self):
        """Once a delta reached the caller, a retry would duplicate it:
        a stream dying mid-body raises ApiError immediately (no backoff
        sleep, no second request)."""
        from python_agent_harness.client import ApiError

        c = make_client(retry_max=3, retry_base_delay=60.0, retry_max_delay=60.0)
        try:
            class DieAfterDelta:
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

            with mock.patch.object(c._http, "stream", return_value=DieAfterDelta()):
                deltas = []
                with self.assertRaises(ApiError):
                    c.chat([Message(role="user", content="hi")], on_delta=deltas.append)
            self.assertEqual(deltas, ["partial"])
        finally:
            c.close()


if __name__ == "__main__":
    unittest.main()
