"""A minimal MCP server speaking raw JSON-RPC over stdio.

Used by test_mcp.py to exercise the harness's MCP client wrapper
end-to-end WITHOUT needing the SDK on the server side (the SDK is only
installed on the client).  Speaks the wire protocol directly:
newline-delimited JSON-RPC, supporting the legacy `initialize`
handshake, the modern `server/discover` probe (answered with
method-not-found so the client falls back to initialize), `tools/list`
and `tools/call` for a few fake tools.
"""

import json
import sys
import threading

_write_lock = threading.Lock()


def _respond(msg: dict) -> None:
    reply = _reply(msg)
    if reply is None:
        return
    with _write_lock:
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()


TOOLS = [
    {
        "name": "echo",
        "description": "Echo the given text back",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "Text to echo"}},
            "required": ["text"],
        },
    },
    {
        "name": "fail",
        "description": "Always reports an error",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "rich",
        "description": "Returns text + image + structured content",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "hang",
        "description": "Never answers (for timeout tests)",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def _reply(msg: dict) -> dict | None:
    method = msg.get("method")
    rid = msg.get("id")
    if rid is None:
        return None  # notification: no reply expected
    params = msg.get("params") or {}
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": rid,
            "result": {
                "protocolVersion": params.get("protocolVersion", "2024-11-05"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake-mcp-server", "version": "1.0.0"},
            },
        }
    if method == "ping":
        return {"jsonrpc": "2.0", "id": rid, "result": {}}
    if method == "server/discover":
        return {
            "jsonrpc": "2.0",
            "id": rid,
            "error": {"code": -32601, "message": "Method not found"},
        }
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}}
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        if name == "echo":
            result = {"content": [{"type": "text", "text": "echo:" + str(args.get("text", ""))}]}
        elif name == "fail":
            result = {"content": [{"type": "text", "text": "boom"}], "isError": True}
        elif name == "rich":
            result = {
                "content": [
                    {"type": "text", "text": "line1"},
                    {"type": "image", "data": "QUJD", "mimeType": "image/png"},
                ],
                "structuredContent": {"count": 1},
            }
        elif name == "hang":
            import time

            time.sleep(60)
            result = {"content": [{"type": "text", "text": "finally"}]}
        else:
            result = {
                "content": [{"type": "text", "text": f"unknown tool {name}"}],
                "isError": True,
            }
        return {"jsonrpc": "2.0", "id": rid, "result": result}
    return {
        "jsonrpc": "2.0",
        "id": rid,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Requests are answered concurrently (like a real MCP server's
        # anyio task per request), so a hanging tools/call never blocks
        # a sibling request.
        threading.Thread(target=_respond, args=(msg,), daemon=True).start()
    return 0


if __name__ == "__main__":
    sys.exit(main())
