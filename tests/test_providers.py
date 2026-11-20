"""Offline provider tests: SDK clients are stubbed, no network, dummy keys."""

from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import openai
import pytest
from anthropic.types import Message
from openai.types.responses import Response as OpenAIResponse

from pipeline.config import MissingApiKey, ModelConfig, Price, load_config
from pipeline.providers import get_live_provider
from pipeline.providers.base import NonRetryableError, RetryableError, checked_params

CLAUDE = ModelConfig("claude-haiku-5-5", "anthropic", True, 4000, {"output_config": {"effort": "low"}}, Price(0.1, 0.5))
GPT = ModelConfig("gpt-6-luna", "openai", True, 4000, {"reasoning": {"effort": "low"}}, Price(0.1, 0.5))


@pytest.fixture(autouse=True)
def dummy_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")


class Recorder:
    def __init__(self, result: Any = None, exc: Exception | None = None) -> None:
        self.result, self.exc, self.kwargs = result, exc, None

    def __call__(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if self.exc:
            raise self.exc
        return self.result


def _http_response(status: int) -> httpx2.Response:
    return httpx2.Response(status, request=httpx2.Request("POST", "https://api.example.invalid/v1"))


def _anthropic_message() -> Message:
    return Message.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-5-5",
            "content": [
                {"type": "thinking", "thinking": "", "signature": "sig"},
                {"type": "text", "text": "```python\ndef f():\n    return 1\n```"},
            ],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 120, "output_tokens": 300, "output_tokens_details": {"thinking_tokens": 200}},
        }
    )


def _openai_response() -> OpenAIResponse:
    return OpenAIResponse.model_validate(
        {
            "id": "resp_1",
            "object": "response",
            "created_at": 0,
            "model": "gpt-6-luna-2026",
            "status": "completed",
            "output": [
                {"type": "reasoning", "id": "rs_1", "summary": []},
                {
                    "type": "message",
                    "id": "m_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "fixed code", "annotations": []}],
                },
            ],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": 100,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens": 50,
                "output_tokens_details": {"reasoning_tokens": 30},
                "total_tokens": 150,
            },
        }
    )


def test_anthropic_request_and_response_mapping() -> None:
    p = get_live_provider(CLAUDE)
    create = Recorder(_anthropic_message())
    p._client = SimpleNamespace(messages=SimpleNamespace(create=create))
    r = p.ask(CLAUDE, "Fix bugs in f.")
    assert create.kwargs == {
        "model": "claude-haiku-5-5",
        "max_tokens": 4000,
        "messages": [{"role": "user", "content": "Fix bugs in f."}],
        "output_config": {"effort": "low"},
    }
    assert r.text == "```python\ndef f():\n    return 1\n```"  # thinking block excluded
    assert (r.input_tokens, r.output_tokens, r.reasoning_tokens) == (120, 300, 200)
    assert r.model_reported == "claude-haiku-5-5" and r.raw_meta["stop_reason"] == "end_turn"


def test_openai_request_and_response_mapping() -> None:
    p = get_live_provider(GPT)
    create = Recorder(_openai_response())
    p._client = SimpleNamespace(responses=SimpleNamespace(create=create))
    r = p.ask(GPT, "Fix bugs in f.")
    assert create.kwargs == {
        "model": "gpt-6-luna",
        "input": "Fix bugs in f.",
        "max_output_tokens": 4000,
        "store": False,
        "reasoning": {"effort": "low"},
    }
    assert r.text == "fixed code"
    assert (r.input_tokens, r.output_tokens, r.reasoning_tokens) == (100, 50, 30)
    assert r.raw_meta["status"] == "completed"


@pytest.mark.parametrize(
    ("status", "cls", "expected"),
    [
        (429, anthropic.RateLimitError, RetryableError),
        (529, anthropic.APIStatusError, RetryableError),
        (500, anthropic.InternalServerError, RetryableError),
        (400, anthropic.BadRequestError, NonRetryableError),
        (401, anthropic.AuthenticationError, NonRetryableError),
        (404, anthropic.NotFoundError, NonRetryableError),
    ],
)
def test_anthropic_error_classification(status: int, cls: type, expected: type) -> None:
    p = get_live_provider(CLAUDE)
    exc = cls("boom", response=_http_response(status), body=None)
    p._client = SimpleNamespace(messages=SimpleNamespace(create=Recorder(exc=exc)))
    with pytest.raises(expected):
        p.ask(CLAUDE, "x")


def test_anthropic_connection_error_is_retryable() -> None:
    p = get_live_provider(CLAUDE)
    exc = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.example.invalid"))
    p._client = SimpleNamespace(messages=SimpleNamespace(create=Recorder(exc=exc)))
    with pytest.raises(RetryableError):
        p.ask(CLAUDE, "x")


@pytest.mark.parametrize(
    ("status", "cls", "expected"),
    [
        (429, openai.RateLimitError, RetryableError),
        (503, openai.InternalServerError, RetryableError),
        (400, openai.BadRequestError, NonRetryableError),
        (403, openai.PermissionDeniedError, NonRetryableError),
    ],
)
def test_openai_error_classification(status: int, cls: type, expected: type) -> None:
    p = get_live_provider(GPT)
    exc = cls("boom", response=_http_response(status), body=None)
    p._client = SimpleNamespace(responses=SimpleNamespace(create=Recorder(exc=exc)))
    with pytest.raises(expected):
        p.ask(GPT, "x")


@pytest.mark.parametrize("bad", [{"temperature": 0}, {"top_p": 0.9}, {"generation": {"top_k": 5}}])
def test_sampling_params_refused(bad: dict) -> None:
    with pytest.raises(NonRetryableError):
        checked_params(ModelConfig("m", "anthropic", True, 10, bad, Price(0, 0)))


def test_sdk_retries_disabled() -> None:
    assert get_live_provider(CLAUDE)._client.max_retries == 0
    assert get_live_provider(GPT)._client.max_retries == 0


def test_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(MissingApiKey):
        get_live_provider(CLAUDE)


def test_config_models_have_known_providers_and_no_sampling_params() -> None:
    for m in load_config().models:
        checked_params(m)
        assert m.provider in {"anthropic", "openai", "gemini"}
