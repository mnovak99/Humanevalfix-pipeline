"""Load the dataset, sanity-check tasks, draw the stratified sample (data/tasks.json)."""

from __future__ import annotations

import ast
import logging
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from pipeline.config import ROOT, Config, RunContext
from pipeline.dataset import file_sha256, load_rows
from pipeline.io import write_json
from pipeline.s05_execute import PASSED, build_program, execute_program

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SanityResult:
    task_id: str
    canonical_status: str
    buggy_status: str

    @property
    def ok(self) -> bool:
        return self.canonical_status == PASSED and self.buggy_status != PASSED

    @property
    def reason(self) -> str:
        reasons = []
        if self.canonical_status != PASSED:
            reasons.append(f"canonical {self.canonical_status}")
        if self.buggy_status == PASSED:
            reasons.append("buggy passed")
        return "; ".join(reasons)


def _id_num(task_id: str) -> int:
    return int(task_id.split("/")[1])


def buggy_code(row: dict) -> str:
    """The code shown to the model (identical in every language)."""
    return row["declaration"] + row["buggy_solution"]


def canonical_code(row: dict) -> str:
    return row["declaration"] + row["canonical_solution"]


def preamble_of(code: str, entry_point: str) -> str:
    """Code without the entry-point function: imports and helpers a model's fix may rely on."""
    tree = ast.parse(code)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == entry_point)
    lines = code.splitlines(keepends=True)
    first = min([fn.lineno] + [d.lineno for d in fn.decorator_list]) - 1
    assert fn.end_lineno is not None
    return "".join(lines[:first] + lines[fn.end_lineno :]).rstrip("\n") + "\n"


def enrich(row: dict) -> dict:
    code = buggy_code(row)
    return {
        **row,
        "buggy_code": code,
        "canonical_code": canonical_code(row),
        "preamble": preamble_of(code, row["entry_point"]),
    }


def sanity_check(rows: list[dict], cfg: Config, tmp_dir: Path) -> list[SanityResult]:
    """Canonical solution must pass and buggy must fail, using exactly the code models see."""

    def check(row: dict) -> SanityResult:
        canon = execute_program(build_program(canonical_code(row), row["test"]), cfg.exec_timeout_s, tmp_dir)
        buggy = execute_program(build_program(buggy_code(row), row["test"]), cfg.exec_timeout_s, tmp_dir)
        return SanityResult(row["task_id"], canon.status, buggy.status)

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
    rows = load_rows(cfg.dataset_path, cfg.text_fields)
    log.info("loaded %d tasks from %s", len(rows), cfg.dataset_path.relative_to(ROOT).as_posix())

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

    strata_counts = Counter(t["bug_type"] for t in tasks)
    meta = {
        "dataset_path": cfg.dataset_path.relative_to(ROOT).as_posix(),
        "dataset_sha256": file_sha256(cfg.dataset_path),
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
        "strata_population": {k: len(v) for k, v in sorted(by_stratum.items())},
        "strata_sample": dict(sorted(strata_counts.items())),
    }
    write_json(ctx.tasks_path, {"meta": meta, "tasks": tasks})
    log.info("wrote %d tasks to %s", len(tasks), ctx.tasks_path)
    log.info("sample by bug_type: %s", meta["strata_sample"])
