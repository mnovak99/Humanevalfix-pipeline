from dataclasses import replace

import pandas as pd
import pytest

from pipeline.config import ModelConfig, Price, load_config
from pipeline.providers.base import NonRetryableError, Response, RetryableError
from pipeline.providers.fake import BEHAVIOURS, FakeProvider, behaviour_for
from pipeline.s03_run_models import (
    FAILED,
    OK,
    Job,
    call_with_retries,
    current_responses,
    estimate_cost,
    make_record,
    plan_jobs,
    prompt_hash,
    settings_hash,
)
from pipeline.s06_aggregate import COLUMNS, aggregate

M1 = ModelConfig("m1", "anthropic", True, 100, {}, Price(1.0, 5.0))
M2 = ModelConfig("m2", "openai", True, 100, {}, Price(0.5, 2.0))
PROMPTS = [
    {"prompt_id": f"Python/{i}::{lang}", "task_id": f"Python/{i}", "language": lang, "prompt": f"p{i}{lang}", "entry_point": "f"}
    for i in (1, 2)
    for lang in ("en", "cs")
]


def _resp(text: str = "ok") -> Response:
    return Response(text, "m", 10, 20, 0, 0.1, {})


PH = {p["prompt_id"]: prompt_hash(p["prompt"]) for p in PROMPTS}


def _rec(pid: str, run: int, status: str, model: ModelConfig = M1, **extra: object) -> dict:
    base = {"prompt_id": pid, "model": model.name, "run": run, "status": status,
            "settings_hash": settings_hash(model), "prompt_hash": PH[pid]}
    return {**base, **extra}


def test_plan_jobs_resume_skips_ok_and_retries_failed() -> None:
    existing = [
        _rec("Python/1::en", 1, OK),
        _rec("Python/1::en", 2, FAILED),
        _rec("Python/1::cs", 1, FAILED),
        _rec("Python/1::cs", 1, OK),  # later record wins
    ]
    jobs = plan_jobs(PROMPTS, [M1, M2], 2, existing)
    keys = {j.key for j in jobs}
    assert len(jobs) == 4 * 2 * 2 - 2
    assert ("Python/1::en", "m1", 1) not in keys and ("Python/1::cs", "m1", 1) not in keys
    assert ("Python/1::en", "m1", 2) in keys


def test_plan_jobs_reruns_calls_made_with_other_settings() -> None:
    old = replace(M1, params={"output_config": {"effort": "medium"}})
    new = replace(M1, params={"output_config": {"effort": "low"}})
    assert settings_hash(old) != settings_hash(new)
    existing = [_rec("Python/1::en", 1, OK, model=old)]
    assert ("Python/1::en", "m1", 1) in {j.key for j in plan_jobs(PROMPTS, [new], 1, existing)}
    assert ("Python/1::en", "m1", 1) not in {j.key for j in plan_jobs(PROMPTS, [old], 1, existing)}
    # Records from before settings/prompts were tracked count as unknown: rerun.
    legacy = [{"prompt_id": "Python/1::en", "model": "m1", "run": 1, "status": OK}]
    assert ("Python/1::en", "m1", 1) in {j.key for j in plan_jobs(PROMPTS, [new], 1, legacy)}


def test_plan_jobs_reruns_when_prompt_text_changed() -> None:
    existing = [_rec("Python/1::en", 1, OK, prompt_hash=prompt_hash("an older template"))]
    assert ("Python/1::en", "m1", 1) in {j.key for j in plan_jobs(PROMPTS, [M1], 1, existing)}


def test_current_responses_drops_older_configurations() -> None:
    existing = [
        _rec("Python/1::en", 1, OK, prompt_hash="stale"),
        _rec("Python/2::en", 1, OK),
        {**_rec("Python/2::en", 1, OK), "prompt_id": "Python/9::en"},  # no longer a prompt
    ]
    current = current_responses(PROMPTS, [M1], existing)
    assert set(current) == {("Python/2::en", "m1", 1)}


def test_record_stores_settings_and_prompt_hash() -> None:
    rec = make_record(Job(PROMPTS[0], M1, 1), {"status": OK}, 1, dry_run=True)
    assert rec["settings"] == {"provider": "anthropic", "max_tokens": 100, "params": {}}
    assert rec["settings_hash"] == settings_hash(M1)
    assert rec["prompt_hash"] == PH["Python/1::en"]


def test_cost_estimate() -> None:
    cfg = load_config()
    jobs = [Job(PROMPTS[0], M1, 1), Job(PROMPTS[1], M1, 1), Job(PROMPTS[0], M2, 1)]
    costs = estimate_cost(jobs, cfg)
    tok = cfg.cost_estimate_tokens
    assert costs["m1"][0] == 2
    assert costs["m1"][1] == pytest.approx(2 * (tok.input * 1.0 + tok.output * 5.0) / 1e6)


class Flaky:
    def __init__(self, failures: int, exc: type[Exception] = RetryableError) -> None:
        self.failures, self.exc, self.calls = failures, exc, 0

    def ask(self, model_cfg: ModelConfig, prompt: str) -> Response:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc("boom")
        return _resp()


def test_retry_then_success() -> None:
    p = Flaky(2)
    result, attempts = call_with_retries(p, Job(PROMPTS[0], M1, 1), 6, wait_initial_s=0)
    assert result["status"] == OK and attempts == 3


def test_retries_exhausted_is_failed_record() -> None:
    p = Flaky(10)
    result, attempts = call_with_retries(p, Job(PROMPTS[0], M1, 1), 6, wait_initial_s=0)
    assert result["status"] == FAILED and attempts == 6 and "retries exhausted" in result["error"]


def test_non_retryable_not_retried() -> None:
    p = Flaky(10, NonRetryableError)
    result, attempts = call_with_retries(p, Job(PROMPTS[0], M1, 1), 6, wait_initial_s=0)
    assert result["status"] == FAILED and attempts == 1 and p.calls == 1


def test_unexpected_exception_is_failed_record() -> None:
    p = Flaky(10, KeyError)
    result, _ = call_with_retries(p, Job(PROMPTS[0], M1, 1), 6, wait_initial_s=0)
    assert result["status"] == FAILED and "KeyError" in result["error"]


def test_fake_provider_is_deterministic_and_covers_behaviours() -> None:
    kinds = {behaviour_for("m1", f"prompt {i}") for i in range(500)}
    assert kinds == {name for _, name in BEHAVIOURS}
    prompt = next(f"q{i}" for i in range(100) if behaviour_for("m1", f"q{i}") == "canonical")
    fake = FakeProvider({prompt: ("def f():\n    return 1\n", "def f():\n    return 0\n", "f")})
    assert fake.ask(M1, prompt) == fake.ask(M1, prompt)
    assert "```python\ndef f():" in fake.ask(M1, prompt).text


def test_aggregate_joins_and_sorts() -> None:
    tasks = [{"task_id": "Python/2", "bug_type": "b"}, {"task_id": "Python/10", "bug_type": "a"}]
    responses, extracted, executions = [], [], []
    for tid in ("Python/10", "Python/2"):
        for lang in ("en", "cs"):
            pid = f"{tid}::{lang}"
            key = {"prompt_id": pid, "model": "m1", "run": 1}
            responses.append({**key, "task_id": tid, "language": lang, "status": OK, "input_tokens": 5,
                              "output_tokens": 7, "reasoning_tokens": None, "latency_s": 0.5})
            usable = lang == "en"
            extracted.append({**key, "task_id": tid, "language": lang,
                              "extraction_status": "ok" if usable else "empty_response",
                              "exact_match_canonical": True if usable else None,
                              "unchanged_from_buggy": False if usable else None})
            executions.append({**key, "exec_status": "passed" if usable else "skipped"})
    df = aggregate(tasks, responses, extracted, executions)
    assert list(df.columns) == COLUMNS
    assert list(df["task_id"]) == ["Python/2", "Python/2", "Python/10", "Python/10"]
    assert df.loc[df.language == "en", "passed"].tolist() == [1, 1]
    assert df.loc[df.language == "cs", "passed"].tolist() == [0, 0]
    assert df.loc[df.language == "en", "exact_match_canonical"].tolist() == [1, 1]
    assert df.loc[df.language == "en", "unchanged_from_buggy"].tolist() == [0, 0]
    assert df.loc[df.language == "cs", "exact_match_canonical"].isna().all()  # no usable code
    assert pd.isna(df["reasoning_tokens"]).all()
    assert df.loc[0, "bug_type"] == "b"
