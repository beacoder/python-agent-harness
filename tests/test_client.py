"""Client streaming tests against the in-process fake OpenAI server."""

import json
import os
import sys
import unittest

from python_agent_harness.client import Client
from python_agent_harness.models import Message

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


if __name__ == "__main__":
    unittest.main()
