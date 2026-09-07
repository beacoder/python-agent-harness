"""OpenAI-compatible chat client built on httpx.

Supports both streaming (default) and non-streaming requests against
any backend speaking the chat-completions protocol (OpenAI, DeepSeek,
Moonshot/Kimi, GLM/Zhipu, Qwen/DashScope, ...).
"""

from __future__ import annotations

import contextlib
import json
import os
import random
import socket as _socket
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx

from . import config
from .models import Message, ToolCall, ToolSpec, Usage

# serializes appends to the shared LLM log file: concurrent sub-agents
# (each with its own client but ONE shared log_path, see Client.clone)
# finish their interactions in parallel, and interleaved write() calls
# would corrupt the JSON stream
_log_lock = threading.Lock()


class ApiError(Exception):
    """Raised when the API call itself fails (network/HTTP)."""


# Grace between shutdown(SHUT_RDWR) and close() of an in-flight socket
# on macOS (see _shutdown_and_close): 50ms proved too tight under load
# (the woken thread can still be parked in select() when the fd is
# released, losing the wakeup).  200ms keeps Ctrl-C responsive while
# making the fd-reuse race vanishingly unlikely.
_ABORT_SHUTDOWN_GRACE_S = 0.2


class RetryableApiError(ApiError):
    """A transient failure (rate limit, server error) safe to retry.

    Carries the server's ``Retry-After`` value (if any) so the retry
    backoff can honor it.  Permanent errors remain a plain ApiError.
    """

    def __init__(self, message: str, retry_after: str | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class AuthExpiredError(ApiError):
    """Raised on auth-expired HTTP status codes.

    By default HTTP 401, but customizable via
    ``config.AUTH_REFRESH_STATUS_CODES`` — some API gateways return 502
    or other codes when the backend auth token has expired.  Handled
    specially in the retry loop: the API key is re-read from
    config/environment (an external process may have refreshed the
    token) and the request is retried once with the new key.  If the
    key hasn't changed, the error is permanent and propagated as a
    plain ApiError.
    """


def _retryable_status(status: int) -> bool:
    """429 and 5xx are transient; every other error is permanent.

    Status codes in ``config.AUTH_REFRESH_STATUS_CODES`` are excluded —
    they are handled as auth-expired, not retried with backoff.
    """
    if status in config.AUTH_REFRESH_STATUS_CODES:
        return False
    return status == 429 or status >= 500


def _error_text(data: dict[str, Any]) -> str | None:
    """A readable message from an API error body, if any.

    Some backends return ``{"error": "message"}`` or
    ``{"error": {"message": "..."}}`` as the JSON payload (or as a
    mid-stream chunk); ``None`` when the payload carries no error.
    """
    err = data.get("error")
    if not err:
        return None
    if isinstance(err, dict):
        return str(err.get("message") or err)
    return str(err)


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
            return min(max(0, secs), max_delay) + random.uniform(0, 0.5)
    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
    return delay + random.uniform(0, delay * 0.3)


def _llm_log_path() -> Path:
    """Return the LLM log file path for a new session."""
    date_str = time.strftime("%Y%m%d")
    session_id = uuid.uuid4().hex[:8]
    log_dir = os.environ.get("LLM_LOG_DIR")
    if log_dir:
        d = Path(log_dir)
        d.mkdir(parents=True, exist_ok=True)
        return d / f"python-agent-harness-{date_str}-{session_id}.json"
    return Path(tempfile.gettempdir()) / f"python-agent-harness-{date_str}-{session_id}.json"


def _log_llm_interaction(
    log_file: Path | None, payload: dict[str, Any], response_msg: Message, usage: Usage
) -> None:
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
                        "arguments": tc.arguments
                        if isinstance(tc.arguments, str)
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

        with _log_lock, open(log_file, "a", encoding="utf-8") as f:
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
        # Windows (Git for Windows, conda, etc.)
        os.path.join(os.environ.get("PROGRAMFILES", ""), "Git", "mingw64", "ssl", "cert.pem"),
        os.path.join(os.environ.get("PROGRAMFILES(X86)", ""), "Git", "mingw64", "ssl", "cert.pem"),
        os.path.join(os.environ.get("CONDA_PREFIX", ""), "ssl", "cert.pem"),
    ):
        if cand and os.path.isfile(cand):
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
        config_path: str | None = None,
        log_path: Path | None = None,
    ) -> None:
        self.base_url = (base_url or config.DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or _default_api_key()
        self.model: str = model or config.DEFAULT_MODEL
        self.timeout = timeout
        self.verify = verify if verify is not None else _resolve_ca_bundle()
        self.retry_max = config.API_RETRY_MAX if retry_max is None else retry_max
        self.retry_base_delay = (
            config.API_RETRY_BASE_DELAY if retry_base_delay is None else retry_base_delay
        )
        self.retry_max_delay = (
            config.API_RETRY_MAX_DELAY if retry_max_delay is None else retry_max_delay
        )
        self._config_path = config_path
        self._http = httpx.Client(timeout=timeout, verify=self.verify)
        # True while the in-flight request was aborted (Ctrl-C): a
        # connection error on an aborted request must NOT be retried —
        # the user asked to stop.  Cleared at the start of each chat()
        # so a fresh turn may retry normally.
        self._aborted = False
        # an explicit log file is inherited by clones so every request
        # of one session (main + all sub-agents) lands in a single log
        self.log_path = (
            log_path
            if log_path is not None
            else (_llm_log_path() if config.LLM_LOG_ENABLED else None)
        )

    @property
    def context_window(self) -> int:
        """Get the context window for this model.

        Resolution order: config-file ``context_windows`` overrides
        (via ``config.get_context_window_for_model``) -> CONTEXT_WINDOWS
        pattern match -> DEFAULT_CONTEXT_WINDOW.  Resolved on every
        access (no caching), so a runtime model switch or config-file
        edit takes effect immediately; a malformed config falls back to
        the default for that access and recovers once the file is fixed.
        """
        try:
            return config.get_context_window_for_model(self.model, config_path=self._config_path)
        except Exception:
            # a malformed context_windows section must not break the
            # loop: use the safe default, retry on the next access
            return config.DEFAULT_CONTEXT_WINDOW

    def close(self) -> None:
        self._http.close()

    def clone(self) -> Client:
        """A fresh Client with identical settings (no shared state).

        Concurrent requests must never share one Client: ``_reset_http``
        and ``abort`` swap and close the underlying httpx pool, and
        ``_aborted`` is per-request flag state — so one request's
        connection failure (or Ctrl-C abort) would tear down a
        sibling's in-flight request on the same client.  Each
        concurrent sub-agent clones its own client (see
        ``Session.run_subagent``), keeping pools and the abort
        flag strictly per-request.  The log file is shared so one
        session's LLM interactions stay in one log.
        """
        return Client(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            timeout=self.timeout,
            verify=self.verify,
            retry_max=self.retry_max,
            retry_base_delay=self.retry_base_delay,
            retry_max_delay=self.retry_max_delay,
            config_path=self._config_path,
            log_path=self.log_path,
        )

    def abort(self) -> None:
        """Abort the in-flight request (called on cancel).

        A blocked ``iter_lines()`` read must be interrupted so the agent
        loop can stop promptly.  Closing the pool alone is NOT enough:
        ``httpx.Client.close()`` from another thread cannot force-close a
        socket whose fd is parked in a kernel ``recv``.  So we first
        reach into every in-flight connection socket and both
        ``shutdown(SHUT_RDWR)`` it (wakes the blocked read on Linux) and
        ``close()`` the raw fd (the reliable wake on macOS/BSD, where
        ``shutdown`` alone may not rouse a parked ``recv``); the woken
        read surfaces as a connection error the loop treats as a cancel.
        We then close the pool and swap in a fresh client for the next
        request.  See ``_abort_inflight_sockets`` for the platform
        details.
        """
        self._aborted = True
        old = self._http
        self._http = httpx.Client(timeout=self.timeout, verify=self.verify)
        with contextlib.suppress(Exception):  # best effort
            _abort_inflight_sockets(old)
        with contextlib.suppress(Exception):  # best effort
            old.close()

    def _reset_http(self) -> None:
        """Replace the httpx client with a fresh instance.

        Called before every connection-error retry (and on exhaustion)
        so a poisoned pool (stale/dead connections) never dooms the
        retry itself or every subsequent request in the session.
        """
        old = self._http
        self._http = httpx.Client(timeout=self.timeout, verify=self.verify)
        with contextlib.suppress(Exception):  # best effort
            old.close()

    def set_timeout(self, timeout: float) -> None:
        """Update the request timeout and recreate the HTTP pool.

        The pool is created once in ``__init__`` with the initial
        timeout, so changing the attribute alone would not affect
        in-flight/future requests.  Recreating the pool makes the new
        timeout apply to subsequent requests immediately (used by
        /model switching).
        """
        self.timeout = timeout
        self._reset_http()

    def _refresh_api_key(
        self,
        cancel_check: Callable[[], bool] | None = None,
        timeout: float = 30.0,
        poll_interval: float = 2.0,
    ) -> bool:
        """Re-read the API key from the config file and environment.

        Called on HTTP 401 (auth expired): an external process (e.g. a
        token refresh script) may need time to write a new key.  This
        method polls the config file / environment every
        ``poll_interval`` seconds for up to ``timeout`` seconds, waiting
        for the key to change.  If a fresh key is found that differs
        from the current one, update ``self.api_key`` and return True
        (the caller should retry).  If the key remains unchanged after
        the timeout, return False (permanent auth failure).

        ``cancel_check`` is polled each iteration so Ctrl-C aborts the
        wait promptly.
        """
        deadline = time.monotonic() + timeout
        while True:
            try:
                settings = config.load_llm_config(self._config_path)
                new_key = settings.get("api_key") or _default_api_key()
            except Exception:  # noqa: BLE001 - config read must not crash
                new_key = _default_api_key()
            if new_key and new_key != self.api_key:
                self.api_key = new_key
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            if cancel_check is not None and cancel_check():
                return False
            time.sleep(min(poll_interval, remaining))

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
        if stream:
            payload["stream_options"] = {"include_usage": True}
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
        on_retry: Callable[[], None] | None = None,
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
        attempts, honoring ``Retry-After`` when present.  A connection
        error always swaps in a fresh httpx client first (``_reset_http``)
        so a dead connection never poisons the retry — and a stream that
        died mid-body IS retried even when deltas already reached the
        callers: the partial stream is discarded on retry (``on_retry``
        lets the caller drop its live text), so nothing is duplicated in
        the returned message.  Other 4xx errors are permanent and fail
        immediately.  ``cancel_check`` (when given) is polled during
        backoff sleeps so an abort lands promptly instead of after the
        full wait.  ``on_retry`` (when given) is invoked right before a
        retry after a connection error, so a UI can clear the partial
        output and show that the request is being restarted.
        """
        payload = self._payload(
            messages,
            tools,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system,
            reasoning_effort=reasoning_effort,
        )
        # a fresh turn may retry connection errors even if a previous
        # in-flight request was aborted (see abort/_aborted)
        self._aborted = False
        usage = Usage()
        emitted = False

        def wrap_delta(chunk: str) -> None:
            nonlocal emitted
            emitted = True
            if on_delta:
                # a presentational sink (live UI) — its failure (e.g. a
                # BrokenPipeError on a closed terminal) must never reach
                # the retry loop below, or it would be mistaken for a
                # transport error and pointlessly re-send the request
                with contextlib.suppress(Exception):  # streaming UI is best effort
                    on_delta(chunk)

        def wrap_tool_call(name: str, call_id: str, fragment: str) -> None:
            nonlocal emitted
            emitted = True
            if on_tool_call:
                with contextlib.suppress(Exception):  # streaming UI is best effort
                    on_tool_call(name, call_id, fragment)

        attempt = 0
        auth_refreshed = False
        while True:
            attempt += 1
            # Track emission per-attempt.  Whether a PRIOR attempt
            # streamed partial text is irrelevant to retrying this one:
            # that partial was already cleared (via on_retry) when the
            # prior attempt failed.  Resetting here lets a transient
            # status (429/5xx) arriving after a dropped partial stream
            # still be retried, consistent with the connection-error
            # branch below.
            emitted = False
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
                # transient status (429 / 5xx): retry with backoff until
                # the attempt budget is exhausted.  If this attempt had
                # already streamed partial text to the caller, clear it
                # first (mirrors the connection-error branch) so the
                # retry never duplicates output.
                if attempt >= self.retry_max:
                    raise
                if emitted and on_retry is not None:
                    on_retry()
                if self._sleep_backoff(attempt, e.retry_after, cancel_check):
                    raise
            except AuthExpiredError as e:
                # Auth-expired status code (401, or any code in
                # config.AUTH_REFRESH_STATUS_CODES): the credential
                # (often a JWT with a short TTL) may have expired.
                # Re-read the API key from the config file / environment
                # — an external token-refresh process may have written a
                # new one.  Poll for up to 30s waiting for the key to
                # change; retry once if it does.  Fail immediately if
                # already refreshed once
                # (prevents infinite loops).
                self._reset_http()
                if auth_refreshed or not self._refresh_api_key(cancel_check):
                    raise ApiError(str(e)) from e
                auth_refreshed = True
                # Key refreshed — retry immediately (no backoff needed,
                # and only one extra attempt regardless of retry_max)
                if emitted and on_retry is not None:
                    on_retry()
                continue
            except (httpx.HTTPError, OSError) as e:
                # connection-level failures: connect errors, timeouts,
                # dropped streams.  ``httpx.HTTPError`` covers everything
                # httpx wraps, but a raw ``OSError`` (ConnectionResetError,
                # BrokenPipeError, ``ssl.SSLError`` — all OSError
                # subclasses) can still leak from the socket layer,
                # notably out of the streaming generator or during SSL
                # teardown.  Such an error MUST be treated the same way:
                # if it escaped uncaught the poisoned connection would
                # stay in the pool and doom every following request in
                # the session (the reported "connection broken -> all
                # later requests fail" symptom).  Swap in a fresh client
                # immediately — a dead connection must not stay in the
                # pool for the retry — then retry the request, even when
                # deltas already reached the caller: the partial stream
                # is discarded on retry (on_retry lets the caller clear
                # its live text), so nothing is duplicated in the stored
                # message.  Only give up once the per-request attempt
                # budget is exhausted — or immediately when the request
                # was aborted (Ctrl-C: the user asked to stop, so a
                # fresh attempt must not be started).
                self._reset_http()
                if self._aborted or attempt >= self.retry_max:
                    raise ApiError(f"network error: {e}") from e
                if on_retry is not None:
                    on_retry()
                if self._sleep_backoff(attempt, None, cancel_check):
                    raise ApiError(f"network error: {e}") from e
            except Exception:
                # safety net for anything unexpected (permanent ApiError
                # from a 4xx, a parsing bug, a raising on_delta callback,
                # ...).  We do NOT retry these — retrying a permanent
                # error just burns the budget with backoff, and retrying
                # a bug masks it as a bogus "network error".  But the
                # request may have died mid-stream, leaving the
                # connection in an indeterminate state, so we still
                # reset the pool before propagating: a poisoned
                # connection must never survive into the next request,
                # whatever the cause.  (BaseException — KeyboardInterrupt
                # / SystemExit — is intentionally not caught here.)
                self._reset_http()
                raise

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
                if resp.status_code in config.AUTH_REFRESH_STATUS_CODES:
                    raise AuthExpiredError(message)
                if _retryable_status(resp.status_code):
                    raise RetryableApiError(message, resp.headers.get("Retry-After"))
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
                # a 200 stream can still end in an error object
                # (no choices); surface it instead of silently
                # returning an empty message
                err = _error_text(data)
                if err:
                    raise ApiError(f"API error: {err}")
                if data.get("usage"):
                    u = data["usage"]
                    usage.input_tokens = int(u.get("prompt_tokens") or u.get("input_tokens") or 0)
                    usage.output_tokens = int(
                        u.get("completion_tokens") or u.get("output_tokens") or 0
                    )
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
                        tc_index, self._slot_for(tc_index, tc, 0), tc, on_tool_call
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
        resp = self._http.post(self._url(), headers=self._headers(stream=False), json=payload)
        if resp.status_code >= 400:
            message = f"API error {resp.status_code}: {resp.text[:500]}"
            if resp.status_code in config.AUTH_REFRESH_STATUS_CODES:
                raise AuthExpiredError(message)
            if _retryable_status(resp.status_code):
                raise RetryableApiError(message, resp.headers.get("Retry-After"))
            raise ApiError(message)
        data = resp.json()
        err = _error_text(data)
        if err:
            raise ApiError(f"API error: {err}")
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
                p.get("text", "") if isinstance(p, dict) else (p if isinstance(p, str) else "")
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
            self._absorb_tool_call(tc_index, self._slot_for(tc_index, tc, i), tc, on_tool_call)
        return content_parts, reasoning_parts, tc_index

    @staticmethod
    def _slot_for(
        tc_index: dict[int, dict[str, Any]],
        tc: dict[str, Any],
        fallback: int,
    ) -> int:
        """The accumulator slot for one tool-call chunk.

        ``index`` is authoritative when it is a usable integer.
        ``tc.get("index", fallback)`` is not enough: a backend may send
        ``"index": null`` (key present, value null), and the None would
        both land as a dict key — blowing up the final ``sorted(tc_index)``
        with a TypeError — and merge unrelated calls into one slot.

        Without a usable index the newest slot continues (a streaming
        call's fragments arrive in order), unless the chunk carries an
        ``id`` that differs from the one already accumulated there: that
        marks a NEW call, and merging would splice two calls together.
        """
        idx = tc.get("index")
        if isinstance(idx, int) and not isinstance(idx, bool):
            return idx
        if not tc_index:
            return fallback
        current = max(tc_index)
        tid = tc.get("id") or ""
        if tid and tc_index[current]["id"] and tid != tc_index[current]["id"]:
            return current + 1
        return current

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
        cancel_check: Callable[[], bool] | None = None,
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
            cancel_check=cancel_check,
        )


def _default_api_key() -> str | None:
    return os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or None


def _extract_socket(conn: object) -> _socket.socket | None:
    """Best-effort extraction of the raw socket from an httpcore connection.

    For direct (non-proxy) connections the socket lives at
    ``conn._connection._network_stream``.  Proxy connections are wrapped
    (ForwardHTTPConnection/TunnelHTTPConnection -> HTTPConnection ->
    HTTP11Connection), so we unwrap up to 4 levels looking for the
    ``_network_stream`` attribute that owns the raw socket.
    """
    try:
        obj: object = conn
        stream = None
        for _ in range(4):
            stream = getattr(obj, "_network_stream", None)
            if stream is not None:
                break
            obj = getattr(obj, "_connection", None)
            if obj is None:
                break
        return stream.get_extra_info("socket") if stream is not None else None
    except Exception:  # noqa: BLE001 - best effort
        return None


def _shutdown_and_close(sock: _socket.socket) -> None:
    """Shutdown then close a socket to wake blocked reads.

    On macOS the socket is in non-blocking mode (httpcore calls
    settimeout for the read timeout), so a blocked recv parks in
    select(); closing the fd before that select has processed the
    shutdown wakeup can lose the wakeup and leave the read parked
    forever.  Worse, ``abort()`` swaps in a fresh httpx client right
    after, so a new socket could reuse the fd number before the
    reader observed the EOF.  The grace sleep delays the fd release
    so the woken thread can observe the EOF on the old fd first.

    On Windows, ``shutdown(SHUT_RDWR)`` on the client socket causes
    ``WinError 10058`` on the server side when it tries to read,
    producing noisy BrokenPipeError tracebacks.  Skipping shutdown
    and just closing the fd is sufficient — Windows ``close()``
    triggers a TCP RST that reliably wakes the blocked read.
    """
    if sys.platform != "win32":
        with contextlib.suppress(OSError):
            sock.shutdown(_socket.SHUT_RDWR)
    if sys.platform == "darwin":
        # macOS only; on Linux shutdown() alone reliably wakes the
        # blocked recv (measured: <200ms unblock, no sleeps needed).
        time.sleep(_ABORT_SHUTDOWN_GRACE_S)
    with contextlib.suppress(OSError):
        sock.close()


def _abort_inflight_sockets(client: httpx.Client) -> None:
    """Wake any blocked stream reads on every live pool socket.

    Reaches through httpx/httpcore internals (transport -> pool ->
    connection -> network stream -> raw socket) and, for each live
    connection, both ``shutdown(SHUT_RDWR)`` AND ``close()`` the raw
    socket.

    Both steps are needed for cross-platform reliability when another
    thread is already blocked in ``recv``:

    - On Linux, ``shutdown(SHUT_RDWR)`` reliably wakes the blocked
      ``recv`` (turning it into a clean EOF / connection error), while
      httpx's own ``close()`` cannot — it can't force-close a socket
      whose fd is parked in a kernel read.
    - On macOS/BSD (notably CI's macos-latest on py3.11), ``shutdown``
      from another thread is NOT guaranteed to wake a ``recv`` already
      parked in the kernel; the read stays blocked until the fd itself
      is closed.  ``socket.close()`` closes the fd directly and does
      wake it, so we always follow the shutdown with an explicit
      close.

    Closing the fd out from under httpcore is safe here: ``abort()``
    swaps in a fresh pool immediately and closes the old one under
    ``contextlib.suppress`` right after this call, so a double-close is
    harmless.  Every step is best-effort; any failure is ignored.
    """
    pools: list[object] = []
    base_pool = getattr(getattr(client, "_transport", None), "_pool", None)
    if base_pool is not None:
        pools.append(base_pool)

    if sys.platform == "darwin":
        # macOS only: with proxy env vars (HTTP_PROXY/HTTPS_PROXY) set,
        # httpx routes requests through proxy transports registered in
        # client._mounts (pools are httpcore.HTTPProxy), leaving the
        # base transport pool above empty.  On Linux this never happens
        # in CI and the code path above is sufficient, so this extra
        # walk is deliberately restricted to macOS.
        mounts = getattr(client, "_mounts", None)
        if isinstance(mounts, dict):
            for transport in mounts.values():
                if transport is not None:
                    pool = getattr(transport, "_pool", None)
                    if pool is not None:
                        pools.append(pool)

    for pool in pools:
        for conn in getattr(pool, "_connections", None) or ():
            sock = _extract_socket(conn)
            if sock is not None:
                _shutdown_and_close(sock)


def _iter_sse(lines: Iterator[str]) -> Iterator[str]:
    """Yield SSE data payloads from a line iterator."""
    for line in lines:
        line = line.strip()
        if line.startswith("data:"):
            yield line[len("data:") :].strip()
