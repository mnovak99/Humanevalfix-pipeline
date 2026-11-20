"""Provider interface shared by all LLM backends."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from pipeline.config import ModelConfig


@dataclass(frozen=True)
class Response:
    text: str
    model_reported: str | None
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    latency_s: float
    raw_meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class RetryableError(Exception):
    """Transient failure (429, 5xx, network); the caller retries with backoff."""


class NonRetryableError(Exception):
    """Permanent failure (bad request, auth, ...); logged as a failed record."""


class Provider(Protocol):
    def ask(self, model_cfg: ModelConfig, prompt: str) -> Response: ...
