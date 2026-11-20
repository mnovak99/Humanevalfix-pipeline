import importlib
from pathlib import Path

import pytest

from pipeline.__main__ import STEPS, build_parser
from pipeline.config import RunContext, load_config


def test_config_loads() -> None:
    cfg = load_config()
    assert cfg.languages[0] == "en"
    assert cfg.pilot_n_tasks <= cfg.n_tasks
    assert all(m.provider in {"anthropic", "openai", "gemini"} for m in cfg.models)
    assert "temperature" not in str([m.params for m in cfg.models])


def test_mode_paths_are_separate(tmp_path: Path) -> None:
    cfg = load_config()
    modes = [
        RunContext(cfg, pilot=p, dry_run=d, root=tmp_path) for p in (False, True) for d in (False, True)
    ]
    assert len({c.responses_path for c in modes}) == 4
    assert len({c.results_path for c in modes}) == 4
    assert RunContext(cfg, pilot=True, root=tmp_path).tasks_path.name == "tasks_pilot.json"
    assert RunContext(cfg, pilot=True, root=tmp_path).prompts_path.name == "prompts_pilot.jsonl"
    assert RunContext(cfg, root=tmp_path).n_tasks == cfg.n_tasks


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "--step", "01", "--pilot"],
        ["run", "--step", "03", "--dry-run"],
        ["run", "--all", "--pilot", "--dry-run"],
        ["mock-translations"],
    ],
)
def test_cli_parses(argv: list[str]) -> None:
    build_parser().parse_args(argv)


def test_all_steps_importable() -> None:
    assert sorted(STEPS) == ["01", "02", "03", "04", "05", "06"]
    for module in STEPS.values():
        assert callable(importlib.import_module(module).run)
