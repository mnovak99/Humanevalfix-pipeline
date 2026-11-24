from pathlib import Path

import pytest

from pipeline.__main__ import build_parser
from pipeline.config import RunContext, UnknownVariant, apply_variant, load_config
from pipeline.providers.base import checked_params
from pipeline.s03_run_models import settings_hash
from pipeline.ui.runner import pipeline_args

CFG = load_config()


def _model(cfg, name):
    return next(m for m in cfg.models if m.name == name)


def test_noreason_variant_settings() -> None:
    cfg = apply_variant(CFG, "noreason")
    haiku, gpt = _model(cfg, "claude-haiku-5-5"), _model(cfg, "gpt-6-luna")
    assert haiku.params == {"thinking": {"type": "disabled"}, "output_config": {"effort": "low"}}
    assert gpt.params == {"reasoning": {"effort": "none"}}
    # Everything except params is unchanged; the default config itself is untouched.
    assert (haiku.max_tokens, haiku.price_per_mtok, haiku.enabled) == (
        _model(CFG, "claude-haiku-5-5").max_tokens,
        _model(CFG, "claude-haiku-5-5").price_per_mtok,
        True,
    )
    assert _model(CFG, "gpt-6-luna").params == {"reasoning": {"effort": "low"}}
    for m in cfg.models:
        checked_params(m)  # no sampling parameters


def test_haiku_effort_allows_disabled_thinking() -> None:
    # Haiku 5.5 rejects thinking {type: disabled} at effort xhigh/max.
    haiku = _model(apply_variant(CFG, "noreason"), "claude-haiku-5-5")
    assert haiku.params["output_config"]["effort"] in {"low", "medium", "high"}


def test_variant_changes_settings_hash() -> None:
    noreason = apply_variant(CFG, "noreason")
    for name in ("claude-haiku-5-5", "gpt-6-luna"):
        assert settings_hash(_model(CFG, name)) != settings_hash(_model(noreason, name))


def test_no_variant_and_unknown_variant() -> None:
    assert apply_variant(CFG, None) is CFG
    with pytest.raises(UnknownVariant):
        apply_variant(CFG, "nope")


def test_variant_outputs_are_separate_but_inputs_shared(tmp_path: Path) -> None:
    base = RunContext(CFG, pilot=True, root=tmp_path)
    var = RunContext(apply_variant(CFG, "noreason"), pilot=True, variant="noreason", root=tmp_path)
    var_dry = RunContext(apply_variant(CFG, "noreason"), pilot=True, variant="noreason", dry_run=True, root=tmp_path)
    assert var.tasks_path == base.tasks_path and var.prompts_path == base.prompts_path
    assert var.responses_path.name == "responses_pilot_noreason.jsonl"
    assert var.results_path.name == "results_pilot_noreason.csv"
    assert var_dry.results_path.name == "results_pilot_noreason_dry.csv"
    assert len({base.results_path, var.results_path, var_dry.results_path}) == 3


def test_confirm_treats_end_of_input_as_no(monkeypatch: pytest.MonkeyPatch) -> None:
    from pipeline import s03_run_models

    monkeypatch.setattr(s03_run_models.sys.stdin, "isatty", lambda: True, raising=False)

    def eof(_prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    assert s03_run_models.confirm("Proceed?", assume_yes=False) is False
    assert s03_run_models.confirm("Proceed?", assume_yes=True) is True


def test_cli_and_ui_pass_the_variant() -> None:
    args = build_parser().parse_args(["run", "--all", "--pilot", "--variant", "noreason"])
    assert args.variant == "noreason"
    assert build_parser().parse_args(["smoke", "--variant", "noreason"]).variant == "noreason"
    assert pipeline_args("all", pilot=True, variant="noreason", yes=True) == [
        "run", "--all", "--pilot", "--variant", "noreason", "--yes",
    ]
