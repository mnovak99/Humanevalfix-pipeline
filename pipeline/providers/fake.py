"""Deterministic fake provider for --dry-run: no network, exercises every downstream status."""

from __future__ import annotations

import hashlib

from pipeline.config import ModelConfig
from pipeline.providers.base import NonRetryableError, Response

# Behaviour buckets over hash % 100 (upper bounds, exclusive).
BEHAVIOURS = [
    (55, "canonical"),  # fixed code in a ```python block -> passed
    (65, "buggy"),  # returns the bug unchanged -> failed/error
    (75, "canonical_bare"),  # code without a fence -> no_code_block
    (83, "syntax_error"),
    (90, "function_missing"),
    (95, "empty"),
    (100, "api_error"),  # simulated non-retryable provider error
]


def behaviour_for(model: str, prompt: str) -> str:
    h = int(hashlib.sha256(f"{model}\n{prompt}".encode("utf-8")).hexdigest(), 16) % 100
    return next(name for bound, name in BEHAVIOURS if h < bound)


class FakeProvider:
    def __init__(self, solutions: dict[str, tuple[str, str, str]]) -> None:
        """solutions: prompt text -> (canonical_code, buggy_code, entry_point)."""
        self._solutions = solutions

    def ask(self, model_cfg: ModelConfig, prompt: str) -> Response:
        canonical, buggy, entry = self._solutions[prompt]
        kind = behaviour_for(model_cfg.name, prompt)
        text = {
            "canonical": f"Here is the fixed function:\n\n```python\n{canonical.strip()}\n```\n",
            "buggy": f"```python\n{buggy.strip()}\n```",
            "canonical_bare": canonical.strip() + "\n",
            "syntax_error": f"```python\ndef {entry}(:\n    return\n```",
            "function_missing": "```python\ndef something_else():\n    return 42\n```",
            "empty": "",
        }.get(kind)
        if text is None:
            raise NonRetryableError("fake provider: simulated invalid request")
        return Response(
            text=text,
            model_reported=f"fake/{model_cfg.name}",
            input_tokens=len(prompt) // 4,
            output_tokens=len(text) // 4,
            reasoning_tokens=0,
            latency_s=0.0,
            raw_meta={"fake_behaviour": kind},
        )
