"""Call enabled models for every prompt and run; append each result to responses.jsonl."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime

from tenacity import (
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)
from tqdm import tqdm

from pipeline.config import Config, ModelConfig, RunContext
from pipeline.io import append_jsonl, read_json, read_jsonl
from pipeline.providers import NonRetryableError, Provider, RetryableError, get_live_provider
from pipeline.providers.base import checked_params
from pipeline.providers.fake import FakeProvider

log = logging.getLogger(__name__)

OK, FAILED = "ok", "failed"


class LiveRunRefused(RuntimeError):
    pass


@dataclass(frozen=True)
class Job:
    prompt: dict
    model: ModelConfig
    run: int

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.prompt["prompt_id"], self.model.name, self.run)


def response_key(rec: dict) -> tuple[str, str, int]:
    return (rec["prompt_id"], rec["model"], int(rec["run"]))


def latest_by_key(records: list[dict]) -> dict[tuple[str, str, int], dict]:
    """Later records win (a failed call can be retried by a later run)."""
    return {response_key(r): r for r in records}


def model_settings(model: ModelConfig) -> dict:
    """Everything about a model call that can change its output (besides the prompt)."""
    return {"provider": model.provider, "max_tokens": model.max_tokens, "params": model.params}


def settings_hash(model: ModelConfig) -> str:
    blob = json.dumps(model_settings(model), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def prompt_hash(prompt_text: str) -> str:
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:12]


def plan_jobs(prompts: list[dict], models: list[ModelConfig], runs: int, existing: list[dict]) -> list[Job]:
    """All prompt x model x run jobs without a successful record for the same prompt and settings.

    A record made with a different prompt text (templates changed) or different model
    settings (e.g. effort) does not count as done, so results never mix configurations;
    the new record supersedes it downstream.
    """
    current_settings = {m.name: settings_hash(m) for m in models}
    current_prompt = {p["prompt_id"]: prompt_hash(p["prompt"]) for p in prompts if p.get("prompt")}
    done = {
        k
        for k, r in latest_by_key(existing).items()
        if r["status"] == OK
        and r.get("settings_hash") == current_settings.get(r["model"])
        and r.get("prompt_hash") == current_prompt.get(r["prompt_id"])
    }
    jobs = [Job(p, m, run) for p in prompts for m in models for run in range(1, runs + 1)]
    return [j for j in jobs if j.key not in done]


def current_responses(prompts: list[dict], models: list[ModelConfig], existing: list[dict]) -> dict[tuple[str, str, int], dict]:
    """Latest record per key, keeping only those made with the current prompt and settings.

    Records for other prompt texts, models or settings (older configurations) are left in
    responses.jsonl as history but never reach extraction or results.
    """
    current_settings = {m.name: settings_hash(m) for m in models}
    current_prompt = {p["prompt_id"]: prompt_hash(p["prompt"]) for p in prompts}
    return {
        k: r
        for k, r in latest_by_key(existing).items()
        if r.get("settings_hash") == current_settings.get(r["model"])
        and r.get("prompt_hash") == current_prompt.get(r["prompt_id"])
    }


def estimate_cost(jobs: list[Job], cfg: Config) -> dict[str, tuple[int, float]]:
    per_model: dict[str, tuple[int, float]] = {}
    tok = cfg.cost_estimate_tokens
    for j in jobs:
        price = j.model.price_per_mtok
        cost = (tok.input * price.input + tok.output * price.output) / 1e6
        n, c = per_model.get(j.model.name, (0, 0.0))
        per_model[j.model.name] = (n + 1, c + cost)
    return per_model


def confirm(question: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        log.error("no interactive terminal for confirmation; rerun with --yes")
        return False
    try:
        return input(f"{question} [y/N] ").strip().lower() in {"y", "yes"}
    except EOFError:  # e.g. stdin is the NUL device, which Windows reports as a terminal
        log.error("no answer to the confirmation; rerun with --yes")
        return False


def call_with_retries(
    provider: Provider, job: Job, max_attempts: int, wait_initial_s: float = 2.0
) -> tuple[dict, int]:
    """Return (response fields or error, attempts used)."""
    attempts = 0
    try:
        for attempt in Retrying(
            retry=retry_if_exception_type(RetryableError),
            wait=wait_exponential_jitter(initial=wait_initial_s, max=60, jitter=wait_initial_s),
            stop=stop_after_attempt(max_attempts),
            reraise=False,
        ):
            with attempt:
                attempts = attempt.retry_state.attempt_number
                resp = provider.ask(job.model, job.prompt["prompt"])
        return {"status": OK, "error": None, **resp.to_dict()}, attempts
    except RetryError as exc:
        err = exc.last_attempt.exception()
        return {"status": FAILED, "error": f"retries exhausted: {type(err).__name__}: {err}"}, attempts
    except NonRetryableError as exc:
        return {"status": FAILED, "error": f"{type(exc).__name__}: {exc}"}, attempts
    except Exception as exc:  # unexpected: log as failed record instead of killing the run
        log.exception("unexpected error for %s", job.key)
        return {"status": FAILED, "error": f"{type(exc).__name__}: {exc}"}, attempts


def make_record(job: Job, result: dict, attempts: int, dry_run: bool) -> dict:
    empty = {
        "text": None,
        "model_reported": None,
        "input_tokens": None,
        "output_tokens": None,
        "reasoning_tokens": None,
        "latency_s": None,
        "raw_meta": {},
    }
    return {
        "prompt_id": job.prompt["prompt_id"],
        "task_id": job.prompt["task_id"],
        "language": job.prompt["language"],
        "model": job.model.name,
        "provider": "fake" if dry_run else job.model.provider,
        "run": job.run,
        "settings": model_settings(job.model),
        "settings_hash": settings_hash(job.model),
        "prompt_hash": prompt_hash(job.prompt["prompt"]),
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "attempts": attempts,
        **empty,
        **result,
    }


def execute_jobs(
    jobs: list[Job],
    provider_for: Callable[[ModelConfig], Provider],
    ctx: RunContext,
) -> tuple[int, int]:
    lock = threading.Lock()
    n_ok = n_failed = 0

    def work(job: Job) -> dict:
        result, attempts = call_with_retries(provider_for(job.model), job, ctx.config.retry_max_attempts)
        rec = make_record(job, result, attempts, ctx.dry_run)
        with lock:  # append immediately so an interrupted run loses nothing
            append_jsonl(ctx.responses_path, rec)
        return rec

    workers = 1 if ctx.dry_run else ctx.config.api_workers
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, j) for j in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="model calls"):
            rec = fut.result()
            if rec["status"] == OK:
                n_ok += 1
            else:
                n_failed += 1
                log.debug("failed %s/%s/run%s: %s", rec["prompt_id"], rec["model"], rec["run"], rec["error"])
    return n_ok, n_failed


def fake_provider_for(ctx: RunContext, prompts: list[dict]) -> FakeProvider:
    tasks = {t["task_id"]: t for t in read_json(ctx.tasks_path)["tasks"]}
    solutions = {
        p["prompt"]: (
            tasks[p["task_id"]]["canonical_code"],
            tasks[p["task_id"]]["buggy_code"],
            p["entry_point"],
        )
        for p in prompts
    }
    return FakeProvider(solutions)


def run(ctx: RunContext) -> None:
    cfg = ctx.config
    if not ctx.prompts_path.exists():
        raise FileNotFoundError(f"{ctx.prompts_path} not found; run step 02 first")
    prompts = read_jsonl(ctx.prompts_path)
    models = cfg.enabled_models
    if not models:
        raise LiveRunRefused("no enabled models in config.yaml")

    existing = read_jsonl(ctx.responses_path)
    jobs = plan_jobs(prompts, models, cfg.runs_per_prompt, existing)
    total = len(prompts) * len(models) * cfg.runs_per_prompt
    log.info("%d/%d calls already done, %d to go -> %s", total - len(jobs), total, len(jobs), ctx.responses_path)
    if not jobs:
        return

    if ctx.dry_run:
        fake = fake_provider_for(ctx, prompts)
        provider_for: Callable[[ModelConfig], Provider] = lambda _m: fake  # noqa: E731
    else:
        for m in models:
            try:
                checked_params(m)
            except NonRetryableError as exc:
                raise LiveRunRefused(str(exc)) from exc
        # Build clients first: a missing key fails here, before the user confirms anything.
        live = {m.name: get_live_provider(m) for m in models}
        costs = estimate_cost(jobs, cfg)
        tok = cfg.cost_estimate_tokens
        log.info("LIVE run, cost estimate (%d input / %d output tokens per call):", tok.input, tok.output)
        for name, (n, cost) in costs.items():
            log.info("  %-28s %5d calls  ~$%.4f", name, n, cost)
        log.info("  %-28s %5d calls  ~$%.4f", "TOTAL", len(jobs), sum(c for _, c in costs.values()))
        if not confirm("Proceed with paid API calls?", ctx.yes):
            raise LiveRunRefused("live run not confirmed")
        provider_for = lambda m: live[m.name]  # noqa: E731

    t0 = time.perf_counter()
    n_ok, n_failed = execute_jobs(jobs, provider_for, ctx)
    log.info("done in %.1fs: %d ok, %d failed (failed calls are retried on the next run)", time.perf_counter() - t0, n_ok, n_failed)
