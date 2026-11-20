"""Call enabled models for every prompt and run; append to responses.jsonl."""

from __future__ import annotations

from pipeline.config import RunContext


def run(ctx: RunContext) -> None:
    raise NotImplementedError("s03_run_models is not implemented yet")
