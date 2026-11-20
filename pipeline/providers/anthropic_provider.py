"""Anthropic Messages API provider (official `anthropic` SDK)."""

from __future__ import annotations

import time

import anthropic

from pipeline.config import ModelConfig, get_api_key
from pipeline.providers.base import (
    NonRetryableError,
    Response,
    RetryableError,
    checked_params,
    is_retryable_status,
)

REQUEST_TIMEOUT_S = 300.0


class AnthropicProvider:
    def __init__(self) -> None:
        # Retries are handled by s03 (tenacity); the SDK's own retries would multiply them.
        self._client = anthropic.Anthropic(
            api_key=get_api_key("anthropic"), max_retries=0, timeout=REQUEST_TIMEOUT_S
        )

    def ask(self, model_cfg: ModelConfig, prompt: str) -> Response:
        # `params` are passed through as request fields, e.g. {"output_config": {"effort": "low"}}.
        params = checked_params(model_cfg)
        t0 = time.perf_counter()
        try:
            msg = self._client.messages.create(
                model=model_cfg.name,
                max_tokens=model_cfg.max_tokens,
                messages=[{"role": "user", "content": prompt}],
                **params,
            )
        except (anthropic.APIConnectionError, anthropic.RateLimitError) as exc:  # includes timeouts
            raise RetryableError(f"{type(exc).__name__}: {exc}") from exc
        except anthropic.APIStatusError as exc:
            err = RetryableError if is_retryable_status(exc.status_code) else NonRetryableError
            raise err(f"{type(exc).__name__} {exc.status_code}: {exc.message}") from exc
        latency = time.perf_counter() - t0

        text = "".join(b.text for b in msg.content if b.type == "text")
        usage = msg.usage
        details = usage.output_tokens_details
        stop_details = msg.stop_details.model_dump() if msg.stop_details else None
        return Response(
            text=text,
            model_reported=msg.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,  # includes thinking tokens
            reasoning_tokens=details.thinking_tokens if details else None,
            latency_s=latency,
            raw_meta={
                "id": msg.id,
                "request_id": msg._request_id,
                "stop_reason": msg.stop_reason,
                "stop_details": stop_details,
                "block_types": [b.type for b in msg.content],
                "cache_read_input_tokens": usage.cache_read_input_tokens,
                "service_tier": usage.service_tier,
            },
        )
