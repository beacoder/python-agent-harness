"""Token estimation and calibration, ported from gptel-agent-harness.el.

Estimates are heuristic (Latin ~4 chars/token, CJK ~2 chars/token);
a calibration factor derived from API-reported input tokens is applied
to reduce drift.
"""

from __future__ import annotations

import json
import re

from . import config


def is_cjk_char(c: str) -> bool:
    """Return True if C is a CJK or full-width character."""
    cp = ord(c)
    return (
        0x3000 <= cp <= 0x9FFF  # CJK + kana + punctuation
        or 0xF900 <= cp <= 0xFAFF  # CJK compat ideographs
        or 0xFF00 <= cp <= 0xFFEF  # full-width forms
        or 0x20000 <= cp <= 0x2FA1F  # CJK extensions B-F
    )


# The same ranges as `is_cjk_char`, as a single compiled character class.
# Scanning for CJK runs in C (the regex engine) instead of a per-character
# Python loop, so large payloads count CJK chars much faster.
_CJK_RE = re.compile(r"[\u3000-\u9fff\uf900-\ufaff\uff00-\uffef\U00020000-\U0002fa1f]")


def estimate_tokens(text: str) -> int:
    """Estimate tokens in TEXT: Latin ~4 chars/token, CJK ~2 chars/token."""
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    latin = len(text) - cjk
    return round(latin / 4.0 + cjk / 2.0)


def context_window_for(model: str) -> int:
    """Return the context window for MODEL, or a safe fallback.

    Delegates to ``config._match_context_window`` (fnmatch over
    CONTEXT_WINDOWS, first match wins, case-insensitive); unknown
    models get DEFAULT_CONTEXT_WINDOW.
    """
    matched = config._match_context_window(model)
    if matched is not None:
        return matched
    return config.DEFAULT_CONTEXT_WINDOW


class TokenCalibrator:
    """Calibration factor: actual_tokens / estimated_tokens.

    Updated after each response using the API-reported input token
    count.  Applied to future estimations to reduce drift.  Clamped
    to [CALIBRATION_MIN, CALIBRATION_MAX] to avoid pathological values.
    """

    def __init__(self) -> None:
        self.factor = 1.0
        self.last_raw_estimate: int | None = None

    def update(self, actual_input: int | None) -> None:
        raw = self.last_raw_estimate
        if actual_input is None or actual_input <= 0 or raw is None or raw <= 0:
            return
        ratio = actual_input / float(raw)
        ratio = max(config.CALIBRATION_MIN, min(config.CALIBRATION_MAX, ratio))
        self.factor = ratio

    def calibrate(self, estimated: int) -> int:
        return round(estimated * self.factor)


def payload_text(system: object, messages: list[dict], tools: list[dict]) -> str:
    """Serialize the full prompt payload into one plain-text buffer."""
    buf: list[str] = []
    if isinstance(system, str):
        buf.append(system)
    elif isinstance(system, dict) and isinstance(system.get("parts"), list):
        for part in system["parts"]:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                buf.append(part["text"])
    elif isinstance(system, list):
        for s in system:
            if isinstance(s, str):
                buf.append(s)
            elif isinstance(s, dict) and isinstance(s.get("text"), str):
                buf.append(s["text"])
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            buf.append(content)
        elif isinstance(content, list):
            for p in content:
                if isinstance(p, str):
                    buf.append(p)
                elif isinstance(p, dict):
                    for key in ("thinking", "text", "arguments"):
                        v = p.get(key)
                        if isinstance(v, str):
                            buf.append(v)
                            break
        reasoning = msg.get("reasoning_content")
        if isinstance(reasoning, str):
            buf.append(reasoning)
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            name = fn.get("name") if isinstance(fn, dict) else None
            args = fn.get("arguments") if isinstance(fn, dict) else None
            if isinstance(name, str):
                buf.append(name)
            if isinstance(args, str):
                buf.append(args)
    for tool in tools:
        buf.append(json.dumps(tool, separators=(",", ":")))
    return "\n".join(buf)


def estimate_payload_tokens(system: object, messages: list[dict], tools: list[dict]) -> int:
    return estimate_tokens(payload_text(system, messages, tools))
