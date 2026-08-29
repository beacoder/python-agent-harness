"""Model capability discovery via LLM provider APIs."""

from __future__ import annotations

import httpx


class ModelInfo:
    """Information about an LLM model including context window."""

    def __init__(
        self,
        model_id: str,
        context_window: int | None = None,
        max_output_tokens: int | None = None,
        supports_reasoning: bool = False,
    ):
        self.model_id = model_id
        self.context_window = context_window
        self.max_output_tokens = max_output_tokens
        self.supports_reasoning = supports_reasoning

    def __repr__(self) -> str:
        return f"ModelInfo({self.model_id}, ctx={self.context_window})"


class ModelDiscoveryError(RuntimeError):
    """A TRANSIENT failure talking to the provider's /models endpoint.

    Raised for network errors, timeouts, rate limits (429) and server
    errors (5xx): the provider may have the information once it
    recovers, so callers must NOT cache the fallback — the next access
    should retry discovery.  Permanent absence of information (403/404/
    405, model not listed, no context field) is NOT an error:
    ``fetch_model_info`` returns ``ModelInfo(context_window=None)`` for
    those.
    """


def _get_models_endpoint(base_url: str) -> str:
    """Return the models endpoint URL for the given base URL."""
    # OpenAI-compatible endpoints expose /models (with or without /v1)
    return f"{base_url.rstrip('/')}/models"


def fetch_model_info(
    base_url: str,
    api_key: str,
    model: str,
    timeout: float = 30.0,
) -> ModelInfo:
    """Fetch model information from the provider's models endpoint.

    Queries the /models endpoint (OpenAI-compatible) to get the
    context_window field if available.

    Raises:
        ModelDiscoveryError: transient failure (network, timeout, 429,
            5xx, malformed body) — the provider may recover, callers
            should retry later.

    Returns:
        ModelInfo with context_window if discovered; ModelInfo with
        context_window=None when the provider permanently has no
        information (no usable endpoint, model not listed, or no
        context field).
    """
    url = _get_models_endpoint(base_url)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()

            # Find the specific model in the list
            models = data.get("data", [])
            for m in models:
                if m.get("id") == model:
                    # Try common field names for context window
                    ctx = (
                        m.get("context_window")
                        or m.get("context_size")
                        or m.get("max_context_length")
                        or m.get("max_input_tokens")
                    )
                    if ctx is not None:
                        # Convert to int if string
                        if isinstance(ctx, str):
                            try:
                                ctx = int(ctx)
                            except ValueError:
                                ctx = None
                        return ModelInfo(
                            model_id=model,
                            context_window=int(ctx) if ctx else None,
                            max_output_tokens=m.get("max_output_tokens"),
                            supports_reasoning=bool(m.get("supports_reasoning", False)),
                        )

            # Model not found in list: permanent absence of info.
            return ModelInfo(model_id=model, context_window=None)

    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        if status in (403, 404, 405):
            # No usable /models endpoint: permanent absence of info.
            return ModelInfo(model_id=model, context_window=None)
        # 429 / 5xx: transient — the provider may recover.
        raise ModelDiscoveryError(str(e)) from e
    except httpx.HTTPError as e:
        # Connection errors, timeouts: transient.
        raise ModelDiscoveryError(str(e)) from e
    except Exception as e:
        # Malformed responses etc.: transient.
        raise ModelDiscoveryError(str(e)) from e
