import pandas as pd
import pytest

from pipeline.config import ModelConfig, Price, load_config
from pipeline.providers.base import NonRetryableError, Response, RetryableError
from pipeline.providers.fake import BEHAVIOURS, FakeProvider, behaviour_for
from pipeline.s03_run_models import (
    FAILED,
    OK,
    Job,
    LiveRunRefused,
    call_with_retries,
    check_not_mock,
    estimate_cost,
    plan_jobs,
)
from pipeline.s06_aggregate import aggregate

M1 = ModelConfig("m1", "anthropic", True, 100, {}, Price(1.0, 5.0))
M2 = ModelConfig("m2", "openai", True, 100, {}, Price(0.5, 2.0))
PROMPTS = [
    {"prompt_id": f"Python/{i}::{lang}", "task_id": f"Python/{i}", "language": lang, "prompt": f"p{i}{lang}", "entry_point": "f"}
    for i in (1, 2)
    for lang in ("en", "cs")
]


def _resp(text: str = "ok") -> Response:
    return Response(text, "m", 10, 20, 0, 0.1, {})


def test_plan_jobs_resume_skips_ok_and_retries_failed() -> None:
    existing = [
        {"prompt_id": "Python/1::en", "model": "m1", "run": 1, "status": OK},
        {"prompt_id": "Python/1::en", "model": "m1", "run": 2, "status": FAILED},
        {"prompt_id": "Python/1::cs", "model": "m1", "run": 1, "status": FAILED},
        {"prompt_id": "Python/1::cs", "model": "m1", "run": 1, "status": OK},  # later success wins
    ]
    jobs = plan_jobs(PROMPTS, [M1, M2], 2, existing)
    keys = {j.key for j in jobs}
    assert len(jobs) == 4 * 2 * 2 - 2
    assert ("Python/1::en", "m1", 1) not in keys and ("Python/1::cs", "m1", 1) not in keys
    assert ("Python/1::en", "m1", 2) in keys


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


def test_live_refuses_mock_prompts() -> None:
    check_not_mock([{**PROMPTS[0], "mock_translation": False}])
    with pytest.raises(LiveRunRefused):
        check_not_mock([{**PROMPTS[1], "mock_translation": True}])
    with pytest.raises(LiveRunRefused):
        check_not_mock([{**PROMPTS[1], "prompt": "[CS] Oprav chyby", "mock_translation": False}])


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
            extracted.append({**key, "extraction_status": "ok"})
            executions.append({**key, "exec_status": "passed" if lang == "en" else "failed"})
    df = aggregate(tasks, responses, extracted, executions)
    assert list(df["task_id"]) == ["Python/2", "Python/2", "Python/10", "Python/10"]
    assert df.loc[df.language == "en", "passed"].tolist() == [1, 1]
    assert df.loc[df.language == "cs", "passed"].tolist() == [0, 0]
    assert pd.isna(df["reasoning_tokens"]).all()
    assert df.loc[0, "bug_type"] == "b"
