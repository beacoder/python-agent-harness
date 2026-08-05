"""Fake OpenAI-compatible server for smoke-testing the client."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        stream = body.get("stream", False)
        if stream:
            chunks = [
                {"choices": [{"delta": {"role": "assistant", "content": "Hello"}}]},
                {"choices": [{"delta": {"content": " world"}}]},
                {"choices": [{"delta": {"tool_calls": [
                    {"index": 0, "id": "call_1",
                     "function": {"name": "Read",
                                  "arguments": '{"file_path": "/tmp/x.py"}'}}
                ]}}]},
                {"choices": [{"delta": {}}], "usage": {"prompt_tokens": 12, "completion_tokens": 3}},
            ]
            data = "".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n"
        else:
            data = json.dumps({
                "choices": [{"message": {"role": "assistant", "content": "sync reply"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            })
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
