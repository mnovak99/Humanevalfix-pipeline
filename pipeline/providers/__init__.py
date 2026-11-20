"""LLM provider implementations (anthropic, openai, gemini, fake)."""

from __future__ import annotations

from pipeline.config import ModelConfig
from pipeline.providers.base import NonRetryableError, Provider, Response, RetryableError

__all__ = ["NonRetryableError", "Provider", "Response", "RetryableError", "get_live_provider"]


def get_live_provider(model_cfg: ModelConfig) -> Provider:
    raise NotImplementedError(f"provider {model_cfg.provider!r} is implemented in Phase 4")
