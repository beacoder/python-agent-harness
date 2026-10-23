#!/usr/bin/env python3
"""TCP bridge from the Docker bridge gateway to a loopback-only proxy.

On this WSL host the only working egress is an HTTP proxy bound to
127.0.0.1:1080 (the Windows-side proxy, shared through WSL's mirrored
networking). Containers cannot reach 127.0.0.1:1080: inside a container that
address is the container itself. This bridge listens on the Docker bridge
gateway (172.17.0.1:3128) and forwards every connection to 127.0.0.1:1080, so
build steps and containers can use the proxy as http://172.17.0.1:3128.

It is a plain TCP relay: the client speaks HTTP proxy to this process and the
bytes are forwarded verbatim, CONNECT tunnels included.
"""

from __future__ import annotations

import contextlib
import os
import socket
import sys
import threading

LISTEN_HOST = os.environ.get("BRIDGE_LISTEN_HOST", "0.0.0.0")
LISTEN_PORT = int(os.environ.get("BRIDGE_LISTEN_PORT", "3128"))
TARGET_HOST = os.environ.get("BRIDGE_TARGET_HOST", "127.0.0.1")
TARGET_PORT = int(os.environ.get("BRIDGE_TARGET_PORT", "1080"))
BUFSIZE = 65536


def pump(src: socket.socket, dst: socket.socket) -> None:
    try:
        while True:
            chunk = src.recv(BUFSIZE)
            if not chunk:
                break
            dst.sendall(chunk)
    except OSError:
        pass
    finally:
        for sock in (src, dst):
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)


def handle(client: socket.socket) -> None:
    try:
        upstream = socket.create_connection((TARGET_HOST, TARGET_PORT), timeout=30)
    except OSError as exc:
        print(f"connect to {TARGET_HOST}:{TARGET_PORT} failed: {exc}", file=sys.stderr)
        client.close()
        return
    upstream.settimeout(None)
    client.settimeout(None)
    threading.Thread(target=pump, args=(client, upstream), daemon=True).start()
    pump(upstream, client)


def main() -> int:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((LISTEN_HOST, LISTEN_PORT))
    server.listen(128)
    print(f"bridging {LISTEN_HOST}:{LISTEN_PORT} -> {TARGET_HOST}:{TARGET_PORT}", flush=True)
    while True:
        client, _ = server.accept()
        threading.Thread(target=handle, args=(client,), daemon=True).start()


if __name__ == "__main__":
    raise SystemExit(main())
