"""Join step outputs into results.csv and print a short sanity summary."""

from __future__ import annotations

import logging

import pandas as pd

from pipeline.config import RunContext
from pipeline.io import read_json, read_jsonl
from pipeline.s03_run_models import latest_by_key

log = logging.getLogger(__name__)

KEY = ["prompt_id", "model", "run"]
COLUMNS = [
    "task_id",
    "bug_type",
    "language",
    "model",
    "run",
    "extraction_status",
    "exec_status",
    "passed",
    "exact_match_canonical",
    "unchanged_from_buggy",
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "latency_s",
]


def aggregate(tasks: list[dict], responses: list[dict], extracted: list[dict], executions: list[dict]) -> pd.DataFrame:
    """One row per extracted record (step 04 already dropped records of older configurations)."""
    token_cols = ["input_tokens", "output_tokens", "reasoning_tokens", "latency_s"]
    resp = pd.DataFrame(list(latest_by_key(responses).values()))[KEY + token_cols]
    ext = pd.DataFrame(extracted)[
        KEY + ["task_id", "language", "extraction_status", "exact_match_canonical", "unchanged_from_buggy"]
    ]
    exe = pd.DataFrame(executions)[KEY + ["exec_status"]]
    bug = pd.DataFrame([{"task_id": t["task_id"], "bug_type": t["bug_type"]} for t in tasks])

    df = ext.merge(resp, on=KEY, how="left", validate="one_to_one")
    df = df.merge(exe, on=KEY, how="left", validate="one_to_one")
    df = df.merge(bug, on="task_id", how="left", validate="many_to_one")
    if df["exec_status"].isna().any():
        raise ValueError(f"{int(df['exec_status'].isna().sum())} records have no execution result; rerun step 05")
    df["passed"] = (df["exec_status"] == "passed").astype(int)
    # 1/0, empty when there was no usable code to compare.
    for col in ("exact_match_canonical", "unchanged_from_buggy"):
        df[col] = df[col].map({True: 1, False: 0}).astype("Int64")
    for col in ("input_tokens", "output_tokens", "reasoning_tokens"):
        df[col] = df[col].astype("Int64")
    df = df[COLUMNS].copy()
    df["_num"] = df["task_id"].str.split("/").str[1].astype(int)
    df = df.sort_values(["_num", "language", "model", "run"]).drop(columns="_num")
    return df.reset_index(drop=True)


def summary(df: pd.DataFrame) -> str:
    pass_rate = df.pivot_table(index="model", columns="language", values="passed", aggfunc="mean")
    n = df.pivot_table(index="model", columns="language", values="passed", aggfunc="size")
    extraction = df["extraction_status"].value_counts().sort_index()
    execution = df["exec_status"].value_counts().sort_index()
    lines = [
        f"rows: {len(df)}  tasks: {df['task_id'].nunique()}  runs: {sorted(df['run'].unique().tolist())}",
        "pass rate (model x language):",
        pass_rate.map(lambda x: f"{x:.3f}").to_string(),
        "rows per cell:",
        n.to_string(),
        "extraction status:",
        extraction.to_string(),
        f"extraction failures (not ok/no_code_block): {int((~df['extraction_status'].isin(['ok', 'no_code_block'])).sum())}",
        "execution status:",
        execution.to_string(),
        f"exact match with canonical: {int(df['exact_match_canonical'].sum())}, "
        f"unchanged from buggy: {int(df['unchanged_from_buggy'].sum())} "
        f"(of {int(df['exact_match_canonical'].notna().sum())} rows with usable code)",
    ]
    return "\n".join(lines)


def run(ctx: RunContext) -> None:
    for path in (ctx.responses_path, ctx.extracted_path, ctx.executions_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run the previous steps first")
    df = aggregate(
        read_json(ctx.tasks_path)["tasks"],
        read_jsonl(ctx.responses_path),
        read_jsonl(ctx.extracted_path),
        read_jsonl(ctx.executions_path),
    )
    expected = len(read_jsonl(ctx.prompts_path)) * len(ctx.config.enabled_models) * ctx.config.runs_per_prompt
    if len(df) < expected:
        log.warning("results cover %d of %d planned calls (step 03 incomplete?)", len(df), expected)
    ctx.results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ctx.results_path, index=False, encoding="utf-8", lineterminator="\n")
    log.info("wrote %d rows to %s\n%s", len(df), ctx.results_path, summary(df))
