"""EN-only smoke test of the real APIs: a few prompts, one call per enabled model.

Checks keys, model ids and `params` before a paid experiment run. The prompts are the
English experiment prompts from step 02; everything is written to *_smoke files and never
mixes with research data.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from pipeline import s03_run_models, s04_extract, s05_execute, s06_aggregate
from pipeline.config import Config, RunContext, apply_variant
from pipeline.io import read_json, write_jsonl
from pipeline.s02_build_prompts import PromptRecord, build_prompts

log = logging.getLogger(__name__)


def build_smoke_prompts(tasks: list[dict], config: Config) -> list[PromptRecord]:
    return [r for r in build_prompts(tasks, config) if r.language == "en"]


def run_smoke(
    config: Config, n_tasks: int, yes: bool = False, dry_run: bool = False, variant: str | None = None
) -> RunContext:
    # One run per prompt: this checks connectivity and settings, not variance.
    cfg = replace(apply_variant(config, variant), runs_per_prompt=1)
    ctx = RunContext(cfg, smoke=True, yes=yes, dry_run=dry_run, variant=variant)
    if not ctx.tasks_path.exists():
        raise FileNotFoundError(f"{ctx.tasks_path} not found; run step 01 first")
    tasks = read_json(ctx.tasks_path)["tasks"][:n_tasks]
    prompts = build_smoke_prompts(tasks, config)
    write_jsonl(ctx.prompts_path, (p.to_dict() for p in prompts))
    log.info("smoke test: %d EN prompts x %d models -> %s", len(prompts), len(config.enabled_models), ctx.prompts_path)
    for step in (s03_run_models, s04_extract, s05_execute, s06_aggregate):
        step.run(ctx)
    return ctx
