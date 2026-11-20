from pathlib import Path

from pipeline.config import RunContext, load_config
from pipeline.s02_build_prompts import build_prompts
from pipeline.smoke import build_smoke_prompts
from pipeline.translations import load_and_validate, write_workbook
from tests.test_translations_s02 import IDS, TASKS

CFG = load_config()


def test_smoke_prompts_equal_experiment_en_prompts(tmp_path: Path) -> None:
    path = tmp_path / "translations.xlsx"
    write_workbook(path, TASKS, CFG, mock=True)
    tr = load_and_validate(path, CFG.languages, TASKS, IDS)
    en = [r for r in build_prompts(TASKS, tr, CFG.languages) if r.language == "en"]
    smoke = build_smoke_prompts(TASKS, CFG.instruction_template_en)
    assert smoke == en
    assert not any(p.mock_translation for p in smoke)


def test_smoke_paths_are_separate(tmp_path: Path) -> None:
    normal = RunContext(CFG, root=tmp_path)
    smoke = RunContext(CFG, smoke=True, root=tmp_path)
    smoke_dry = RunContext(CFG, smoke=True, dry_run=True, root=tmp_path)
    assert smoke.tasks_path == normal.tasks_path  # reads the full sampled set
    assert smoke.prompts_path.name == "prompts_smoke.jsonl"
    assert smoke.responses_path.name == "responses_smoke.jsonl"
    assert smoke_dry.results_path.name == "results_smoke_dry.csv"
    assert len({normal.responses_path, smoke.responses_path, smoke_dry.responses_path}) == 3
