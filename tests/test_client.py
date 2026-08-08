"""Client streaming tests against the in-process fake OpenAI server."""

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


def make_client() -> Client:
    srv = serve()
    host, port = srv.server_address
    c = Client(base_url=f"http://{host}:{port}/v1", api_key="test", model="fake")
    c._server = srv  # keep the server alive for the test
    return c


class TestClientStreaming(unittest.TestCase):
    def test_reasoning_content_streamed_and_captured(self):
        """reasoning_content deltas stream normally (on_delta) AND are
        captured on the message so the TUI can collapse them later."""
        c = make_client()
        deltas: list[str] = []
        msg, usage = c.chat(
            [Message(role="user", content="hi")],
            on_delta=deltas.append,
        )
        self.assertEqual("".join(deltas), "thinking hardHello world")
        self.assertEqual(msg.content, "thinking hardHello world")
        self.assertEqual(msg.reasoning, "thinking hard")
        self.assertIn("Hello world", msg.content)
        self.assertEqual(usage.input_tokens, 12)
        c.close()


if __name__ == "__main__":
    unittest.main()
