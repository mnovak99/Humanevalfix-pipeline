"""LLM provider implementations (anthropic, openai, gemini, fake)."""

from __future__ import annotations

from pipeline.config import ModelConfig
from pipeline.providers.base import NonRetryableError, Provider, Response, RetryableError

__all__ = ["NonRetryableError", "Provider", "Response", "RetryableError", "get_live_provider"]


def get_live_provider(model_cfg: ModelConfig) -> Provider:
    """Real provider for a model; SDKs are imported lazily so dry runs never need them."""
    if model_cfg.provider == "anthropic":
        from pipeline.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider()
    if model_cfg.provider == "openai":
        from pipeline.providers.openai_provider import OpenAIProvider

        return OpenAIProvider()
    if model_cfg.provider == "gemini":
        from pipeline.providers.gemini_provider import GeminiProvider

        return GeminiProvider()
    raise ValueError(f"unknown provider {model_cfg.provider!r} for model {model_cfg.name!r}")
