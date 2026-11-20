"""OpenAI Responses API provider (official `openai` SDK)."""

from __future__ import annotations

import time

import openai

from pipeline.config import ModelConfig, get_api_key
from pipeline.providers.base import (
    NonRetryableError,
    Response,
    RetryableError,
    checked_params,
    is_retryable_status,
)

REQUEST_TIMEOUT_S = 300.0


class OpenAIProvider:
    def __init__(self) -> None:
        # Retries are handled by s03 (tenacity); the SDK's own retries would multiply them.
        self._client = openai.OpenAI(api_key=get_api_key("openai"), max_retries=0, timeout=REQUEST_TIMEOUT_S)

    def ask(self, model_cfg: ModelConfig, prompt: str) -> Response:
        # `params` are passed through, e.g. {"reasoning": {"effort": "low"}} -> reasoning=...
        params = checked_params(model_cfg)
        t0 = time.perf_counter()
        try:
            resp = self._client.responses.create(
                model=model_cfg.name,
                input=prompt,
                max_output_tokens=model_cfg.max_tokens,
                store=False,
                **params,
            )
        except (openai.APIConnectionError, openai.RateLimitError) as exc:  # includes timeouts
            raise RetryableError(f"{type(exc).__name__}: {exc}") from exc
        except openai.APIStatusError as exc:
            err = RetryableError if is_retryable_status(exc.status_code) else NonRetryableError
            raise err(f"{type(exc).__name__} {exc.status_code}: {exc.message}") from exc
        latency = time.perf_counter() - t0

        if resp.error is not None:
            raise NonRetryableError(f"response error: {resp.error.code}: {resp.error.message}")
        usage = resp.usage
        return Response(
            text=resp.output_text,
            model_reported=resp.model,
            input_tokens=usage.input_tokens if usage else None,
            output_tokens=usage.output_tokens if usage else None,  # includes reasoning tokens
            reasoning_tokens=usage.output_tokens_details.reasoning_tokens if usage else None,
            latency_s=latency,
            raw_meta={
                "id": resp.id,
                "request_id": getattr(resp, "_request_id", None),
                "status": resp.status,
                "incomplete_reason": resp.incomplete_details.reason if resp.incomplete_details else None,
                "output_types": [item.type for item in resp.output],
            },
        )
