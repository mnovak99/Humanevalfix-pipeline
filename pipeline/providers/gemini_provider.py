"""Google Gemini provider (official `google-genai` SDK). Fallback only; disabled in config."""

from __future__ import annotations

import time

from google import genai
from google.genai import errors, types

from pipeline.config import ModelConfig, get_api_key
from pipeline.providers.base import (
    NonRetryableError,
    Response,
    RetryableError,
    checked_params,
    is_retryable_status,
)


class GeminiProvider:
    def __init__(self) -> None:
        self._client = genai.Client(api_key=get_api_key("gemini"))

    def ask(self, model_cfg: ModelConfig, prompt: str) -> Response:
        # `params` are GenerateContentConfig fields, e.g. {"thinking_config": {"thinking_level": "low"}}.
        params = checked_params(model_cfg)
        config = types.GenerateContentConfig(max_output_tokens=model_cfg.max_tokens, **params)
        t0 = time.perf_counter()
        try:
            resp = self._client.models.generate_content(model=model_cfg.name, contents=prompt, config=config)
        except errors.APIError as exc:
            err = RetryableError if is_retryable_status(exc.code) else NonRetryableError
            raise err(f"{type(exc).__name__} {exc.code}: {exc.message}") from exc
        except (ConnectionError, TimeoutError) as exc:
            raise RetryableError(f"{type(exc).__name__}: {exc}") from exc
        latency = time.perf_counter() - t0

        usage = resp.usage_metadata
        finish = resp.candidates[0].finish_reason if resp.candidates else None
        return Response(
            text=resp.text or "",
            model_reported=resp.model_version,
            input_tokens=usage.prompt_token_count if usage else None,
            output_tokens=usage.candidates_token_count if usage else None,
            reasoning_tokens=usage.thoughts_token_count if usage else None,
            latency_s=latency,
            raw_meta={"finish_reason": str(finish) if finish else None, "response_id": resp.response_id},
        )
