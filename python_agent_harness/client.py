"""OpenAI-compatible chat client built on httpx.

Supports both streaming (default) and non-streaming requests against
any backend speaking the chat-completions protocol (OpenAI, DeepSeek,
Moonshot/Kimi, GLM/Zhipu, Qwen/DashScope, ...).
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import httpx

from . import config
from .models import Message, ToolCall, ToolSpec, Usage


class ApiError(Exception):
    """Raised when the API call itself fails (network/HTTP)."""


class RetryableApiError(ApiError):
    """A transient failure (rate limit, server error) safe to retry.

    Carries the server's ``Retry-After`` value (if any) so the retry
    backoff can honor it.  Permanent errors remain a plain ApiError.
    """

    def __init__(self, message: str, retry_after: str | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def _retryable_status(status: int) -> bool:
    """429 and 5xx are transient; every other error is permanent."""
    return status == 429 or status >= 500


def _retry_delay(
    attempt: int,
    retry_after: str | None,
    base_delay: float,
    max_delay: float,
) -> float:
    """Backoff delay for the failed ATTEMPT (1 = first attempt).

    Computed as ``base_delay`` doubled per attempt, capped at
    ``max_delay``, plus jitter.  A ``Retry-After`` header (seconds)
    from a 429 response wins when present.
    """
    if isinstance(retry_after, str) and retry_after.strip():
        try:
            secs = float(retry_after.strip())
        except ValueError:
            pass
        else:
            return min(secs, max_delay) + random.uniform(0, 0.5)
    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
    return delay + random.uniform(0, delay * 0.3)


def _llm_log_path() -> Path:
    """Return the LLM log file path for a new session."""
    import uuid
    date_str = time.strftime("%Y%m%d")
    session_id = uuid.uuid4().hex[:8]
    log_dir = os.environ.get("LLM_LOG_DIR")
    if log_dir:
        d = Path(log_dir)
        d.mkdir(parents=True, exist_ok=True)
        return d / f"python-agent-harness-{date_str}-{session_id}.json"
    return Path(f"/tmp/python-agent-harness-{date_str}-{session_id}.json")


def _log_llm_interaction(log_file: "Path | None", payload: dict[str, Any], response_msg: "Message", usage: "Usage") -> None:
    """Append an LLM interaction to the log file as pretty-printed JSON."""
    if not log_file:
        return
    try:

        # Build the entry in the same format as the conversation:
        # { "model": ..., "messages": [...all messages including response...] }
        messages = list(payload.get("messages", []))

        # Append the assistant response
        resp: dict[str, Any] = {"role": "assistant"}
        if response_msg.text():
            resp["content"] = response_msg.text()
        if response_msg.tool_calls:
            resp["tool_calls"] = [
                {
                    "type": "function",
                    "id": tc.id,
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments if isinstance(tc.arguments, str)
                        else json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in response_msg.tool_calls
            ]
        messages.append(resp)

        body: dict[str, Any] = {
            "model": payload.get("model", ""),
            "messages": messages,
        }
        if payload.get("tools"):
            body["tools"] = payload["tools"]

        marker: dict[str, Any] = {
            "python-agent-harness": "request body",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(marker, indent=2, ensure_ascii=False) + "\n")
            f.write(json.dumps(body, indent=2, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 - logging must never break the agent
        pass


def _resolve_ca_bundle() -> str | bool:
    """Return a CA bundle path usable for TLS verification, or True (default).

    Python's bundled cert.pem often lacks the internal CA chain,
    so prefer a system CA bundle when one exists.  An explicit
    SSL_CERT_FILE env var wins; otherwise fall back to common system
    bundle locations before letting httpx use its default.
    """
    env = os.environ.get("SSL_CERT_FILE")
    if env and os.path.isfile(env):
        return env
    for cand in (
        "/etc/pki/tls/certs/ca-bundle.crt",
        "/etc/ssl/certs/ca-certificates.crt",
        "/etc/ssl/ca-bundle.pem",
    ):
        if os.path.isfile(cand):
            return cand
    return True


class Client:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 600.0,
        verify: str | bool | None = None,
        retry_max: int | None = None,
        retry_base_delay: float | None = None,
        retry_max_delay: float | None = None,
    ) -> None:
        self.base_url = (base_url or config.DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or _default_api_key()
        self.model = model or config.DEFAULT_MODEL
        self.timeout = timeout
        self.verify = verify if verify is not None else _resolve_ca_bundle()
        self.retry_max = config.API_RETRY_MAX if retry_max is None else retry_max
        self.retry_base_delay = (
            config.API_RETRY_BASE_DELAY if retry_base_delay is None else retry_base_delay
        )
        self.retry_max_delay = (
            config.API_RETRY_MAX_DELAY if retry_max_delay is None else retry_max_delay
        )
        self._http = httpx.Client(timeout=timeout, verify=self.verify)
        self.log_path: Path | None = _llm_log_path() if config.LLM_LOG_ENABLED else None

    def close(self) -> None:
        self._http.close()

    def abort(self) -> None:
        """Abort the in-flight request (called on cancel).

        A blocked ``iter_lines()`` read must be interrupted so the agent
        loop can stop promptly.  Closing the pool alone is NOT enough:
        on Linux, ``close()`` from another thread cannot wake a ``recv``
        that is already blocked in the kernel.  So we first
        ``shutdown(SHUT_RDWR)`` every in-flight connection socket (which
        does wake the blocked read, turning it into a connection error
        the loop treats as a cancel) and then close the pool.  A fresh
        client is swapped in for the next request.
        """
        old = self._http
        self._http = httpx.Client(timeout=self.timeout, verify=self.verify)
        try:
            _abort_inflight_sockets(old)
        except Exception:  # noqa: BLE001 - best effort
            pass
        try:
            old.close()
        except Exception:  # noqa: BLE001 - best effort
            pass

    def _reset_http(self) -> None:
        """Replace the httpx client with a fresh instance.

        Called after connection-level retries are exhausted so a
        poisoned pool (stale/dead connections) does not doom every
        subsequent request in the session.
        """
        old = self._http
        self._http = httpx.Client(timeout=self.timeout, verify=self.verify)
        try:
            old.close()
        except Exception:  # noqa: BLE001 - best effort
            pass

    # -- request plumbing -------------------------------------------------
    def _headers(self, stream: bool = True) -> dict[str, str]:
        h = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
        }
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _payload(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        stream: bool = True,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system: str | None = None,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        msgs = [m.to_api() for m in messages]
        if system:
            msgs = [{"role": "system", "content": system}] + msgs
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": msgs,
            "stream": stream,
        }
        if tools:
            payload["tools"] = [t.to_api() for t in tools]
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
        return payload

    # -- chat --------------------------------------------------------------
    def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        on_delta: Callable[[str], None] | None = None,
        on_tool_call: Callable[[str, str, str], None] | None = None,
        stream: bool = True,
        cancel_check: Callable[[], bool] | None = None,
    ) -> tuple[Message, Usage]:
        """Send a chat request, return (assistant msg, usage).

        With ``stream`` True (default) the response is streamed: on_delta
        is invoked with each text chunk as it arrives and on_tool_call
        (name, id, json_fragment) with each tool-call fragment.  With
        ``stream`` False a single non-streaming POST is used; both
        callbacks fire once per text/tool-call with the complete values,
        so callers (agent loop, TUI) behave identically either way.

        Transient failures (HTTP 429 / 5xx, connection errors) are
        retried with exponential backoff + jitter up to ``retry_max``
        attempts, honoring ``Retry-After`` when present.  A retry only
        happens before any delta has been delivered to the callbacks,
        so streaming output is never duplicated for the caller.  Other
        4xx errors are permanent and fail immediately.  ``cancel_check``
        (when given) is polled during backoff sleeps so an abort lands
        promptly instead of after the full wait.
        """
        payload = self._payload(
            messages, tools, stream=stream, temperature=temperature,
            max_tokens=max_tokens, system=system,
            reasoning_effort=reasoning_effort,
        )
        usage = Usage()
        emitted = False

        def wrap_delta(chunk: str) -> None:
            nonlocal emitted
            emitted = True
            if on_delta:
                on_delta(chunk)

        def wrap_tool_call(name: str, call_id: str, fragment: str) -> None:
            nonlocal emitted
            emitted = True
            if on_tool_call:
                on_tool_call(name, call_id, fragment)

        attempt = 0
        while True:
            attempt += 1
            try:
                if stream:
                    content_parts, reasoning_parts, tc_index = self._stream_response(
                        payload, wrap_delta, wrap_tool_call, usage
                    )
                else:
                    content_parts, reasoning_parts, tc_index = self._sync_response(
                        payload, wrap_delta, wrap_tool_call, usage
                    )
                break
            except RetryableApiError as e:
                if emitted or attempt >= self.retry_max:
                    raise
                if self._sleep_backoff(attempt, e.retry_after, cancel_check):
                    raise
            except httpx.HTTPError as e:
                # connection-level failures: connect errors, timeouts,
                # dropped streams — all transient unless a delta already
                # reached the caller (then a retry would duplicate it)
                if emitted or attempt >= self.retry_max:
                    # Replace the client so a poisoned connection pool
                    # does not doom all subsequent requests in this
                    # session.  Without this, a single network hiccup
                    # can leave the session stuck in a permanent error
                    # state (the dead connection stays in the pool and
                    # keeps getting reused).
                    self._reset_http()
                    raise ApiError(f"network error: {e}") from e
                if self._sleep_backoff(attempt, None, cancel_check):
                    self._reset_http()
                    raise ApiError(f"network error: {e}") from e

        content = "".join(content_parts)
        tool_calls = None
        if tc_index:
            tool_calls = [
                ToolCall(
                    id=tc_index[i]["id"] or f"call_{i}",
                    name=tc_index[i]["name"],
                    arguments=tc_index[i]["arguments"] or "{}",
                )
                for i in sorted(tc_index)
            ]
        msg = Message(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            reasoning="".join(reasoning_parts) or None,
        )
        _log_llm_interaction(self.log_path, payload, msg, usage)
        return msg, usage

    def _sleep_backoff(
        self,
        attempt: int,
        retry_after: str | None,
        cancel_check: Callable[[], bool] | None,
    ) -> bool:
        """Sleep between retries; return True when aborted (cancelled).

        ``attempt`` is the number of the request that just failed (1 =
        first attempt); the delay doubles per attempt, capped, with
        jitter (``Retry-After`` wins for 429s).  When ``cancel_check``
        is given it is polled in small increments so a Ctrl-C lands
        promptly instead of after the full backoff wait.
        """
        deadline = time.monotonic() + _retry_delay(
            attempt, retry_after, self.retry_base_delay, self.retry_max_delay
        )
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            if cancel_check is not None and cancel_check():
                return True
            time.sleep(min(0.25, remaining))

    def _stream_response(
        self,
        payload: dict[str, Any],
        on_delta: Callable[[str], None] | None,
        on_tool_call: Callable[[str, str, str], None] | None,
        usage: Usage,
    ) -> tuple[list[str], list[str], dict[int, dict[str, Any]]]:
        """POST a streaming request and accumulate SSE deltas.

        Returns (content parts, reasoning parts, tool-call index) with
        the same shape as `_sync_response`, so `chat()` assembles the
        final message identically for both modes.
        """
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tc_index: dict[int, dict[str, Any]] = {}

        with self._http.stream(
            "POST", self._url(), headers=self._headers(stream=True), json=payload
        ) as resp:
            if resp.status_code >= 400:
                body = resp.read().decode("utf-8", "replace")
                message = f"API error {resp.status_code}: {body[:500]}"
                if _retryable_status(resp.status_code):
                    raise RetryableApiError(
                        message, resp.headers.get("Retry-After")
                    )
                raise ApiError(message)
            for chunk in _iter_sse(resp.iter_lines()):
                if not chunk:
                    continue
                if chunk == "[DONE]":
                    break
                try:
                    data = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                if data.get("usage"):
                    u = data["usage"]
                    usage.input_tokens = int(u.get("prompt_tokens") or u.get("input_tokens") or 0)
                    usage.output_tokens = int(u.get("completion_tokens") or u.get("output_tokens") or 0)
                choices = data.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                if isinstance(delta.get("content"), str) and delta["content"]:
                    content_parts.append(delta["content"])
                    if on_delta:
                        on_delta(delta["content"])
                if delta.get("reasoning_content"):
                    content_parts.append(delta["reasoning_content"])
                    reasoning_parts.append(delta["reasoning_content"])
                    if on_delta:
                        on_delta(delta["reasoning_content"])
                for tc in delta.get("tool_calls") or []:
                    self._absorb_tool_call(
                        tc_index, tc.get("index", 0), tc, on_tool_call
                    )
        return content_parts, reasoning_parts, tc_index

    def _sync_response(
        self,
        payload: dict[str, Any],
        on_delta: Callable[[str], None] | None,
        on_tool_call: Callable[[str, str, str], None] | None,
        usage: Usage,
    ) -> tuple[list[str], list[str], dict[int, dict[str, Any]]]:
        """POST a non-streaming request and parse the single response.

        Returns the same shape as `_stream_response`; text deltas are
        fired once with the complete values (reasoning first, mirroring
        the streaming order), and tool-call arguments arrive as one
        complete JSON string instead of fragments.
        """
        resp = self._http.post(
            self._url(), headers=self._headers(stream=False), json=payload
        )
        if resp.status_code >= 400:
            message = f"API error {resp.status_code}: {resp.text[:500]}"
            if _retryable_status(resp.status_code):
                raise RetryableApiError(
                    message, resp.headers.get("Retry-After")
                )
            raise ApiError(message)
        data = resp.json()
        u = data.get("usage")
        if u:
            usage.input_tokens = int(u.get("prompt_tokens") or u.get("input_tokens") or 0)
            usage.output_tokens = int(u.get("completion_tokens") or u.get("output_tokens") or 0)
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tc_index: dict[int, dict[str, Any]] = {}
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        reasoning = msg.get("reasoning_content") or ""
        content = msg.get("content") or ""
        # some backends return content as a list of parts (multimodal);
        # normalize to plain text like Message.text() does, so the
        # assembled parts stay strings
        if isinstance(content, list):
            content = "".join(
                p.get("text", "") if isinstance(p, dict)
                else (p if isinstance(p, str) else "")
                for p in content
            )
        if not isinstance(reasoning, str):
            reasoning = ""
        # mirror the streaming order: reasoning first, then the answer
        if reasoning:
            reasoning_parts.append(reasoning)
            content_parts.append(reasoning)
            if on_delta:
                on_delta(reasoning)
        if content:
            content_parts.append(content)
            if on_delta:
                on_delta(content)
        for i, tc in enumerate(msg.get("tool_calls") or []):
            # honor an explicit index when present (some backends mirror
            # the streaming shape); position is the fallback
            self._absorb_tool_call(tc_index, tc.get("index", i), tc, on_tool_call)
        return content_parts, reasoning_parts, tc_index

    @staticmethod
    def _absorb_tool_call(
        tc_index: dict[int, dict[str, Any]],
        idx: int,
        tc: dict[str, Any],
        on_tool_call: Callable[[str, str, str], None] | None,
    ) -> None:
        """Accumulate one tool-call chunk (a streaming delta fragment or
        a complete non-streaming call) into the index at ``idx``."""
        entry = tc_index.setdefault(
            idx,
            {"id": "", "name": "", "arguments": ""},
        )
        fn = tc.get("function") or {}
        entry["id"] += tc.get("id") or ""
        entry["name"] += fn.get("name") or ""
        frag = fn.get("arguments") or ""
        if isinstance(frag, dict):
            frag = json.dumps(frag, ensure_ascii=False)
        entry["arguments"] += frag
        if on_tool_call and frag:
            on_tool_call(entry["name"], entry["id"], frag)

    # -- non-streaming chat -------------------------------------------------
    def chat_sync(
        self,
        messages: list[Message],
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> tuple[Message, Usage]:
        """Non-streaming request; used for compaction, titles, summary."""
        return self.chat(
            messages,
            tools=None,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            stream=False,
        )


def _default_api_key() -> str | None:
    return (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
        or None
    )


def _abort_inflight_sockets(client: httpx.Client) -> None:
    """Wake any blocked stream reads by shutting down live pool sockets.

    Reaches through httpx/httpcore internals (transport -> pool ->
    connection -> network stream -> raw socket) and calls
    ``shutdown(SHUT_RDWR)`` on each live connection.  On Linux this is
    the only reliable way to interrupt a ``recv`` already blocked in the
    kernel from another thread — ``close()`` cannot do it.  Shutting
    down an idle socket is harmless (the pool is closed right after
    anyway); any failure is ignored (best effort).
    """
    import socket as _socket

    pool = getattr(getattr(client, "_transport", None), "_pool", None)
    for conn in getattr(pool, "_connections", None) or ():
        try:
            stream = getattr(getattr(conn, "_connection", None), "_network_stream", None)
            sock = stream.get_extra_info("socket") if stream is not None else None
        except Exception:  # noqa: BLE001 - best effort
            sock = None
        if sock is not None:
            try:
                sock.shutdown(_socket.SHUT_RDWR)
            except OSError:
                pass


def _iter_sse(lines: Iterator[str]) -> Iterator[str]:
    """Yield SSE data payloads from a line iterator."""
    for line in lines:
        line = line.strip()
        if line.startswith("data:"):
            yield line[len("data:"):].strip()
