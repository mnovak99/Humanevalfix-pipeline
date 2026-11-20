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


# Current models reject or ignore sampling parameters; never send them.
FORBIDDEN_PARAMS = {"temperature", "top_p", "top_k"}
# Transient HTTP statuses: timeout, conflict, rate limit, server errors (529 = overloaded).
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


def checked_params(model_cfg: ModelConfig) -> dict[str, Any]:
    """Config passthrough params, refusing sampling parameters at any nesting level."""

    def walk(obj: Any, path: str) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in FORBIDDEN_PARAMS:
                    raise NonRetryableError(f"{model_cfg.name}: params must not set {path}{k}")
                walk(v, f"{path}{k}.")

    walk(model_cfg.params, "")
    return dict(model_cfg.params)


def is_retryable_status(status: int | None) -> bool:
    return status is not None and (status in RETRYABLE_STATUS or status >= 500)
