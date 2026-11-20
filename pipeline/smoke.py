"""EN-only smoke test of the real APIs: a few prompts, one call per enabled model.

Checks keys, model ids and `params` before the experiment, without needing translations.
Prompts are the English experiment prompts (same template as step 02); everything is
written to *_smoke files and never mixes with research data.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from pipeline import s03_run_models, s04_extract, s05_execute, s06_aggregate
from pipeline.config import Config, RunContext
from pipeline.io import read_json, write_jsonl
from pipeline.s02_build_prompts import PromptRecord, build_prompt, prompt_id

log = logging.getLogger(__name__)


def build_smoke_prompts(tasks: list[dict], instruction_en: str) -> list[PromptRecord]:
    return [
        PromptRecord(
            prompt_id=prompt_id(t["task_id"], "en"),
            task_id=t["task_id"],
            language="en",
            prompt=build_prompt(instruction_en, t, t["buggy_code"]),
            entry_point=t["entry_point"],
            mock_translation=False,
        )
        for t in tasks
    ]


def run_smoke(config: Config, n_tasks: int, yes: bool = False, dry_run: bool = False) -> RunContext:
    # One run per prompt: this checks connectivity and settings, not variance.
    ctx = RunContext(replace(config, runs_per_prompt=1), smoke=True, yes=yes, dry_run=dry_run)
    if not ctx.tasks_path.exists():
        raise FileNotFoundError(f"{ctx.tasks_path} not found; run step 01 first")
    tasks = read_json(ctx.tasks_path)["tasks"][:n_tasks]
    prompts = build_smoke_prompts(tasks, config.instruction_template_en)
    write_jsonl(ctx.prompts_path, (p.to_dict() for p in prompts))
    log.info("smoke test: %d EN prompts x %d models -> %s", len(prompts), len(config.enabled_models), ctx.prompts_path)
    for step in (s03_run_models, s04_extract, s05_execute, s06_aggregate):
        step.run(ctx)
    return ctx
