"""Fake OpenAI-compatible server for smoke-testing the client."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

# Overridable non-streaming response body; None -> default "sync reply".
NON_STREAM_RESPONSE: dict | None = None
# Overridable SSE chunk list for streaming responses; None -> the default
# reasoning+content+tool-call script below.
STREAM_CHUNKS: list[dict] | None = None
# Queue of non-streaming response bodies consumed in order; when
# exhausted, NON_STREAM_RESPONSE / the default reply is used.
NON_STREAM_SEQUENCE: list[dict] = []
# Queue of HTTP statuses consumed in order; a non-200 entry makes the
# server respond with that error status instead of a reply (used to
# exercise client retry behavior).  When exhausted, 200 is used.
STATUS_QUEUE: list[int] = []
# Optional Retry-After header (seconds) attached to error responses.
RETRY_AFTER_HEADER: str | None = None
# Every request body received, in order (for asserting payloads).
REQUEST_BODIES: list[dict] = []


def reset_state() -> None:
    global NON_STREAM_RESPONSE
    NON_STREAM_RESPONSE = None
    global STREAM_CHUNKS
    STREAM_CHUNKS = None
    NON_STREAM_SEQUENCE.clear()
    STATUS_QUEUE.clear()
    REQUEST_BODIES.clear()
    global RETRY_AFTER_HEADER
    RETRY_AFTER_HEADER = None


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        REQUEST_BODIES.append(body)
        status = STATUS_QUEUE.pop(0) if STATUS_QUEUE else 200
        if status != 200:
            err = b'{"error": "transient failure"}'
            self.send_response(status)
            if RETRY_AFTER_HEADER is not None:
                self.send_header("Retry-After", RETRY_AFTER_HEADER)
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)
            return
        stream = body.get("stream", False)
        if stream and STREAM_CHUNKS is not None:
            chunks = STREAM_CHUNKS
            data = "".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n"
        elif stream:
            chunks = [
                {"choices": [{"delta": {"role": "assistant", "reasoning_content": "thinking"}}]},
                {"choices": [{"delta": {"reasoning_content": " hard"}}]},
                {"choices": [{"delta": {"role": "assistant", "content": "Hello"}}]},
                {"choices": [{"delta": {"content": " world"}}]},
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "function": {
                                            "name": "Read",
                                            "arguments": '{"file_path": "/tmp/x.py"}',
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                },
                {
                    "choices": [{"delta": {}}],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 3},
                },
            ]
            data = "".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n"
        elif NON_STREAM_SEQUENCE:
            data = json.dumps(NON_STREAM_SEQUENCE.pop(0))
        elif NON_STREAM_RESPONSE is not None:
            data = json.dumps(NON_STREAM_RESPONSE)
        else:
            data = json.dumps(
                {
                    "choices": [{"message": {"role": "assistant", "content": "sync reply"}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2},
                }
            )
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream" if stream else "application/json")
        self.send_header("Content-Length", str(len(data.encode())))
        self.end_headers()
        self.wfile.write(data.encode())

    def log_message(self, *a):
        pass


def serve():
    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


if __name__ == "__main__":
    srv = serve()
    print(srv.server_address[1])
    import time

    time.sleep(30)
