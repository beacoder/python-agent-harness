"""OpenAI-compatible streaming chat client built on httpx.

Supports any backend speaking the chat-completions protocol
(OpenAI, DeepSeek, Moonshot/Kimi, GLM/Zhipu, Qwen/DashScope, ...).
"""

from __future__ import annotations

import json
from typing import Any, Callable, Iterator

import httpx

from . import config
from .models import Message, ToolCall, ToolSpec, Usage


class ApiError(Exception):
    """Raised when the API call itself fails (network/HTTP)."""


class Client:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 600.0,
    ) -> None:
        self.base_url = (base_url or config.DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or _default_api_key()
        self.model = model or config.DEFAULT_MODEL
        self.timeout = timeout
        self._http = httpx.Client(timeout=timeout)
        self._active_response = None  # set while a stream is being read

    def close(self) -> None:
        self._http.close()

    def abort(self) -> None:
        """Abort the in-flight streaming request (called on cancel).

        Closes the active response so the blocking read in the worker
        thread raises immediately; the caller turns that into a clean
        cancellation instead of an error.
        """
        resp = self._active_response
        if resp is not None:
            try:
                resp.close()
            except Exception:  # noqa: BLE001 - best effort
                pass

    # -- request plumbing -------------------------------------------------
    def _headers(self) -> dict[str, str]:
        h = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
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

    # -- streaming chat ----------------------------------------------------
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
    ) -> tuple[Message, Usage]:
        """Send a chat request, stream deltas, return (assistant msg, usage).

        on_delta is invoked with each text chunk as it arrives;
        on_tool_call(name, id, json_fragment) with each tool-call fragment.
        """
        payload = self._payload(
            messages, tools, stream=True, temperature=temperature,
            max_tokens=max_tokens, system=system,
            reasoning_effort=reasoning_effort,
        )
        content_parts: list[str] = []
        tc_index: dict[int, dict[str, Any]] = {}
        usage = Usage()

        try:
            with self._http.stream(
                "POST", self._url(), headers=self._headers(), json=payload
            ) as resp:
                self._active_response = resp
                try:
                    if resp.status_code >= 400:
                        body = resp.read().decode("utf-8", "replace")
                        raise ApiError(
                            f"API error {resp.status_code}: {body[:500]}"
                        )
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
                            if on_delta:
                                on_delta(delta["reasoning_content"])
                        for tc in delta.get("tool_calls") or []:
                            idx = tc.get("index", 0)
                            entry = tc_index.setdefault(
                                idx,
                                {"id": "", "name": "", "arguments": ""},
                            )
                            fn = tc.get("function") or {}
                            entry["id"] += tc.get("id") or ""
                            entry["name"] += fn.get("name") or ""
                            frag = fn.get("arguments") or ""
                            entry["arguments"] += frag
                            if on_tool_call and frag:
                                on_tool_call(entry["name"], entry["id"], frag)
                finally:
                    self._active_response = None
        except httpx.HTTPError as e:
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
        if content or tool_calls:
            return Message(role="assistant", content=content, tool_calls=tool_calls), usage
        return Message(role="assistant", content=""), usage

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
        payload = self._payload(
            messages, None, stream=False, temperature=temperature,
            max_tokens=max_tokens, system=system,
            reasoning_effort=reasoning_effort,
        )
        try:
            resp = self._http.post(
                self._url(), headers=self._headers(), json=payload
            )
            if resp.status_code >= 400:
                raise ApiError(
                    f"API error {resp.status_code}: {resp.text[:500]}"
                )
            data = resp.json()
        except httpx.HTTPError as e:
            raise ApiError(f"network error: {e}") from e
        usage = Usage()
        u = data.get("usage")
        if u:
            usage.input_tokens = int(u.get("prompt_tokens") or u.get("input_tokens") or 0)
            usage.output_tokens = int(u.get("completion_tokens") or u.get("output_tokens") or 0)
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content = msg.get("content") or ""
        return Message(role="assistant", content=content), usage


def _default_api_key() -> str | None:
    return (
        __import__("os").environ.get("OPENAI_API_KEY")
        or __import__("os").environ.get("DEEPSEEK_API_KEY")
        or None
    )


def _iter_sse(lines: Iterator[str]) -> Iterator[str]:
    """Yield SSE data payloads from a line iterator."""
    for line in lines:
        line = line.strip()
        if line.startswith("data:"):
            yield line[len("data:"):].strip()
