"""Load dataset, sanity-check tasks, stratified sample, docstring split, translation template."""

from __future__ import annotations

import ast
import logging
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from pipeline.config import Config, RunContext
from pipeline.docstrings import DocstringError, split_docstring
from pipeline.io import write_json
from pipeline.s05_execute import PASSED, build_program, execute_program
from pipeline.translations import write_workbook

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SanityResult:
    task_id: str
    canonical_status: str
    buggy_status: str
    split_error: str | None

    @property
    def ok(self) -> bool:
        return self.canonical_status == PASSED and self.buggy_status != PASSED and self.split_error is None

    @property
    def reason(self) -> str:
        reasons = []
        if self.canonical_status != PASSED:
            reasons.append(f"canonical {self.canonical_status}")
        if self.buggy_status == PASSED:
            reasons.append("buggy passed")
        if self.split_error:
            reasons.append(f"docstring split: {self.split_error}")
        return "; ".join(reasons)


def _id_num(task_id: str) -> int:
    return int(task_id.split("/")[1])


def load_dataset_rows(cfg: Config) -> list[dict]:
    from datasets import load_dataset

    d = cfg.dataset
    ds = load_dataset(d.name, d.config, split=d.split, revision=d.revision)
    return [dict(r) for r in ds]


def function_names(source: str) -> set[str]:
    return {n.name for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}


def preamble_of(prompt: str, entry_point: str) -> str:
    """Prompt without the entry-point function: imports and helper functions a fix may rely on."""
    tree = ast.parse(prompt)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == entry_point)
    lines = prompt.splitlines(keepends=True)
    first = min([fn.lineno] + [d.lineno for d in fn.decorator_list]) - 1
    assert fn.end_lineno is not None
    return "".join(lines[:first] + lines[fn.end_lineno :]).rstrip("\n") + "\n"


def enrich(row: dict) -> dict:
    """Add derived fields; raises DocstringError if the docstring cannot be split."""
    buggy_code = row["prompt"] + row["buggy_solution"]
    parts = split_docstring(buggy_code, row["entry_point"], function_names(buggy_code))
    return {
        **row,
        "buggy_code": buggy_code,
        "canonical_code": row["prompt"] + row["canonical_solution"],
        "preamble": preamble_of(row["prompt"], row["entry_point"]),
        "docstring_parts": parts.to_dict(),
    }


def sanity_check(rows: list[dict], cfg: Config, tmp_dir: Path) -> list[SanityResult]:
    def check(row: dict) -> SanityResult:
        split_error = None
        try:
            enrich(row)
        except DocstringError as exc:
            split_error = str(exc)
        canon = execute_program(build_program(row["prompt"] + row["canonical_solution"], row["test"]), cfg.exec_timeout_s, tmp_dir)
        buggy = execute_program(build_program(row["prompt"] + row["buggy_solution"], row["test"]), cfg.exec_timeout_s, tmp_dir)
        return SanityResult(row["task_id"], canon.status, buggy.status, split_error)

    with ThreadPoolExecutor(max_workers=cfg.exec_workers) as pool:
        return list(tqdm(pool.map(check, rows), total=len(rows), desc="sanity check"))


def largest_remainder_quotas(sizes: dict[str, int], n: int, rng: random.Random) -> dict[str, int]:
    total = sum(sizes.values())
    if n > total:
        raise ValueError(f"cannot sample {n} tasks from {total} valid tasks")
    exact = {k: n * v / total for k, v in sizes.items()}
    quotas = {k: int(x) for k, x in exact.items()}
    keys = sorted(sizes)
    rng.shuffle(keys)  # random tie-breaking among equal remainders
    keys.sort(key=lambda k: exact[k] - quotas[k], reverse=True)
    for k in keys[: n - sum(quotas.values())]:
        quotas[k] += 1
    return quotas


def stratified_order(task_ids_by_stratum: dict[str, list[str]], n: int, seed: int) -> list[str]:
    """Proportional stratified sample of n ids, ordered so every prefix is ~proportional too.

    Within each stratum ids are shuffled; then items are interleaved by their relative
    position (i + 0.5) / n_h, so the first k items (e.g. the pilot) keep the strata ratios.
    """
    rng = random.Random(seed)
    strata = {k: sorted(v) for k, v in sorted(task_ids_by_stratum.items())}
    quotas = largest_remainder_quotas({k: len(v) for k, v in strata.items()}, n, rng)
    tie = {k: rng.random() for k in strata}
    keyed: list[tuple[float, float, str]] = []
    for k, ids in strata.items():
        chosen = rng.sample(ids, quotas[k])
        keyed += [((i + 0.5) / quotas[k], tie[k], tid) for i, tid in enumerate(chosen)]
    return [tid for _, _, tid in sorted(keyed)]


def run(ctx: RunContext) -> None:
    cfg = ctx.config
    rows = load_dataset_rows(cfg)
    log.info("loaded %d tasks from %s (%s/%s)", len(rows), cfg.dataset.name, cfg.dataset.config, cfg.dataset.split)

    results = sanity_check(rows, cfg, ctx.tmp_dir)
    excluded = [r for r in results if not r.ok]
    log.info("sanity check: %d/%d tasks valid, %d excluded", len(rows) - len(excluded), len(rows), len(excluded))
    for r in excluded:
        log.warning("excluded %s: %s", r.task_id, r.reason)

    valid_ids = {r.task_id for r in results if r.ok}
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row["task_id"] in valid_ids:
            by_stratum[row["bug_type"]].append(row["task_id"])
    order = stratified_order(by_stratum, cfg.n_tasks, cfg.seed)[: ctx.n_tasks]

    by_id = {row["task_id"]: row for row in rows}
    tasks = [enrich(by_id[tid]) for tid in order]

    strata_counts: dict[str, int] = defaultdict(int)
    for t in tasks:
        strata_counts[t["bug_type"]] += 1
    meta = {
        "dataset": vars(cfg.dataset),
        "seed": cfg.seed,
        "n_tasks": len(tasks),
        "pilot": ctx.pilot,
        "n_dataset": len(rows),
        "n_valid": len(valid_ids),
        "excluded": [
            {"task_id": r.task_id, "canonical": r.canonical_status, "buggy": r.buggy_status, "reason": r.reason}
            for r in excluded
        ],
        "buggy_status_counts": dict(sorted(Counter(r.buggy_status for r in results).items())),
        "buggy_timeouts": sorted((r.task_id for r in results if r.buggy_status == "timeout"), key=_id_num),
        "strata_population":{k: len(v) for k, v in sorted(by_stratum.items())},
        "strata_sample": dict(sorted(strata_counts.items())),
    }
    write_json(ctx.tasks_path, {"meta": meta, "tasks": tasks})
    log.info("wrote %d tasks to %s", len(tasks), ctx.tasks_path)
    if ctx.pilot:
        # The pilot is the first pilot_n_tasks rows of the full template; one template avoids confusion.
        log.info("pilot run: translation template not written (use the full one)")
    else:
        write_workbook(ctx.translations_template_path, tasks, cfg)
        log.info("wrote translation template to %s", ctx.translations_template_path)
    log.info("sample by bug_type: %s", meta["strata_sample"])
