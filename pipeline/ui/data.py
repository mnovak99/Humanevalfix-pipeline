"""Read-only views of the pipeline state for the UI (files only, no model calls)."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import pandas as pd

from pipeline.config import API_KEY_ENV, Config, RunContext
from pipeline.dataset import DatasetError, file_sha256, load_rows
from pipeline.io import read_json, read_jsonl
from pipeline.s03_run_models import OK, current_responses, estimate_cost, plan_jobs
from pipeline.validation import ValidationReport, validate


@dataclass
class FileInfo:
    label: str
    path: Path
    rows: int | None
    modified: datetime | None

    @property
    def exists(self) -> bool:
        return self.modified is not None


def _info(label: str, path: Path, rows: int | None = None) -> FileInfo:
    if not path.exists():
        return FileInfo(label, path, None, None)
    return FileInfo(label, path, rows, datetime.fromtimestamp(path.stat().st_mtime))


def _jsonl_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def step_files(ctx: RunContext) -> list[FileInfo]:
    tasks_rows = len(read_json(ctx.tasks_path)["tasks"]) if ctx.tasks_path.exists() else None
    results_rows = len(pd.read_csv(ctx.results_path)) if ctx.results_path.exists() else None
    return [
        _info("01 tasks", ctx.tasks_path, tasks_rows),
        _info("02 prompts", ctx.prompts_path, _jsonl_rows(ctx.prompts_path)),
        _info("03 responses", ctx.responses_path, _jsonl_rows(ctx.responses_path)),
        _info("04 extracted", ctx.extracted_path, _jsonl_rows(ctx.extracted_path)),
        _info("05 executions", ctx.executions_path, _jsonl_rows(ctx.executions_path)),
        _info("06 results", ctx.results_path, results_rows),
    ]


@dataclass
class CallProgress:
    planned: int
    ok: int
    failed: int

    @property
    def remaining(self) -> int:
        return self.planned - self.ok


def call_progress(ctx: RunContext) -> CallProgress:
    """Done = successful with the current model settings (same rule as resume in step 03)."""
    cfg = ctx.config
    n_prompts = _jsonl_rows(ctx.prompts_path) or ctx.n_tasks * len(cfg.languages)
    planned = n_prompts * len(cfg.enabled_models) * cfg.runs_per_prompt
    remaining = cost_estimate(ctx).calls
    prompts = read_jsonl(ctx.prompts_path)
    current = current_responses(prompts, cfg.enabled_models, read_jsonl(ctx.responses_path))
    failed = sum(1 for r in current.values() if r["status"] != OK)
    return CallProgress(planned, planned - remaining, failed)


@dataclass
class CostEstimate:
    calls: int
    per_model: dict[str, tuple[int, float]]

    @property
    def total(self) -> float:
        return sum(c for _, c in self.per_model.values())


def cost_estimate(ctx: RunContext, n_prompts: int | None = None) -> CostEstimate:
    """Remaining calls and estimated cost; uses prompts.jsonl when it exists, else n_prompts."""
    cfg = ctx.config
    if ctx.prompts_path.exists() and n_prompts is None:
        prompts = read_jsonl(ctx.prompts_path)
    else:
        count = n_prompts if n_prompts is not None else ctx.n_tasks * len(cfg.languages)
        prompts = [
            {"prompt_id": f"planned-{i}", "task_id": "", "language": "", "prompt": "", "entry_point": ""}
            for i in range(count)
        ]
    jobs = plan_jobs(prompts, cfg.enabled_models, cfg.runs_per_prompt, read_jsonl(ctx.responses_path))
    return CostEstimate(len(jobs), estimate_cost(jobs, cfg))


@dataclass
class DatasetStatus:
    path: Path
    load_error: str | None = None
    n_rows: int = 0
    sha256: str | None = None
    scope: str = ""  # which tasks the report covers
    report: ValidationReport | None = None

    @property
    def ok(self) -> bool:
        return self.load_error is None and self.report is not None and self.report.ok


def dataset_status(cfg: Config, pilot: bool) -> DatasetStatus:
    """Validate the translations of the selected task set (all 164 rows before step 01 ran)."""
    st = DatasetStatus(cfg.dataset_path)
    try:
        rows = load_rows(cfg.dataset_path, cfg.text_fields)
    except DatasetError as exc:
        st.load_error = str(exc)
        return st
    st.n_rows, st.sha256 = len(rows), file_sha256(cfg.dataset_path)
    ctx = RunContext(cfg, pilot=pilot)
    if ctx.tasks_path.exists():
        tasks = read_json(ctx.tasks_path)["tasks"]
        st.scope = f"{len(tasks)} sampled tasks ({ctx.tasks_path.name})"
    else:
        tasks = rows
        st.scope = f"all {len(rows)} rows (run step 01 to sample)"
    st.report = validate(tasks, cfg.text_fields)
    return st


def smoke_estimate(cfg: Config, n_tasks: int, variant: str | None = None) -> CostEstimate:
    """`cfg` must already have the variant applied."""
    from pipeline.smoke import build_smoke_prompts

    ctx = RunContext(replace(cfg, runs_per_prompt=1), smoke=True, variant=variant)
    tasks = read_json(ctx.tasks_path)["tasks"][:n_tasks] if ctx.tasks_path.exists() else []
    prompts = [p.to_dict() for p in build_smoke_prompts(tasks, cfg)]
    jobs = plan_jobs(prompts, ctx.config.enabled_models, 1, read_jsonl(ctx.responses_path))
    return CostEstimate(len(jobs), estimate_cost(jobs, ctx.config))


def missing_api_keys(cfg: Config) -> list[str]:
    """Env var names of missing keys for enabled models (presence only, values never read out)."""
    names = {API_KEY_ENV[m.provider] for m in cfg.enabled_models}
    return sorted(n for n in names if not os.environ.get(n, "").strip())


def load_results(ctx: RunContext) -> pd.DataFrame | None:
    return pd.read_csv(ctx.results_path) if ctx.results_path.exists() else None


def pass_rate_table(df: pd.DataFrame) -> pd.DataFrame:
    table = df.pivot_table(index="model", columns="language", values="passed", aggfunc="mean")
    table.columns.name = None
    return table


def record_details(ctx: RunContext, task_id: str, language: str, model: str, run: int) -> dict:
    """Prompt, response, extracted code and execution result for one results row."""
    pid = f"{task_id}::{language}"
    key = (pid, model, int(run))

    def find(rows: list[dict]) -> dict:
        return next((r for r in reversed(rows) if (r["prompt_id"], r["model"], int(r["run"])) == key), {})

    prompt = next((p for p in read_jsonl(ctx.prompts_path) if p["prompt_id"] == pid), {})
    return {
        "prompt": prompt.get("prompt"),
        "response": find(read_jsonl(ctx.responses_path)),
        "extracted": find(read_jsonl(ctx.extracted_path)),
        "execution": find(read_jsonl(ctx.executions_path)),
    }
