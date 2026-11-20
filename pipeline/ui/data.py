"""Read-only views of the pipeline state for the UI (files only, no model calls)."""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

import pandas as pd

from pipeline.config import API_KEY_ENV, Config, RunContext
from pipeline.io import read_json, read_jsonl
from pipeline.s03_run_models import OK, estimate_cost, latest_by_key, plan_jobs
from pipeline.translations import (
    STATUSES,
    TranslationError,
    is_mock_file,
    doc_col,
    load_and_validate,
)


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
    failed = sum(1 for r in latest_by_key(read_jsonl(ctx.responses_path)).values() if r["status"] != OK)
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
class TranslationStatus:
    exists: bool
    is_mock: bool = False
    status_counts: dict[str, int] = field(default_factory=dict)
    errors: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def usable_live(self) -> bool:
        return self.exists and not self.is_mock and self.errors is None


def translation_status(cfg: Config, pilot: bool) -> TranslationStatus:
    path = cfg.translations_path
    if not path.exists():
        return TranslationStatus(False)
    sheets = pd.read_excel(path, sheet_name=None, dtype=str, keep_default_na=False)
    df = sheets.get("docstrings", pd.DataFrame())
    counts = Counter(df["status"].str.strip().str.lower()) if "status" in df else Counter()
    st = TranslationStatus(
        exists=True,
        is_mock=is_mock_file(path, cfg.languages),
        status_counts={s: counts.get(s, 0) for s in STATUSES} | {k: v for k, v in counts.items() if k not in STATUSES},
    )
    ctx = RunContext(cfg, pilot=pilot)
    full = RunContext(cfg)
    if not ctx.tasks_path.exists():
        st.errors = f"{ctx.tasks_path.name} not found; run step 01 first"
        return st
    tasks = read_json(ctx.tasks_path)["tasks"]
    known = {t["task_id"] for t in read_json(full.tasks_path)["tasks"]} if full.tasks_path.exists() else set()
    try:
        st.warnings = load_and_validate(path, cfg.languages, tasks, known).warnings
    except TranslationError as exc:
        st.errors = str(exc)
    return st


def smoke_estimate(cfg: Config, n_tasks: int) -> CostEstimate:
    from pipeline.smoke import build_smoke_prompts

    ctx = RunContext(replace(cfg, runs_per_prompt=1), smoke=True)
    tasks = read_json(ctx.tasks_path)["tasks"][:n_tasks] if ctx.tasks_path.exists() else []
    prompts = [p.to_dict() for p in build_smoke_prompts(tasks, cfg.instruction_template_en)]
    jobs = plan_jobs(prompts, ctx.config.enabled_models, 1, read_jsonl(ctx.responses_path))
    return CostEstimate(len(jobs), estimate_cost(jobs, ctx.config))


def missing_api_keys(cfg: Config) -> list[str]:
    """Env var names of missing keys for enabled models (presence only, values never read out)."""
    names = {API_KEY_ENV[m.provider] for m in cfg.enabled_models}
    return sorted(n for n in names if not os.environ.get(n, "").strip())


def translation_columns(cfg: Config) -> list[str]:
    return [doc_col(lang) for lang in cfg.languages]


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
