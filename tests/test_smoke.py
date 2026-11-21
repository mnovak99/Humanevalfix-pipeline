from pathlib import Path

from pipeline.config import RunContext, load_config

CFG = load_config()


def test_smoke_paths_are_separate(tmp_path: Path) -> None:
    normal = RunContext(CFG, root=tmp_path)
    smoke = RunContext(CFG, smoke=True, root=tmp_path)
    smoke_dry = RunContext(CFG, smoke=True, dry_run=True, root=tmp_path)
    assert smoke.tasks_path == normal.tasks_path  # reads the full sampled set
    assert smoke.prompts_path.name == "prompts_smoke.jsonl"
    assert smoke.responses_path.name == "responses_smoke.jsonl"
    assert smoke_dry.results_path.name == "results_smoke_dry.csv"
    assert len({normal.responses_path, smoke.responses_path, smoke_dry.responses_path}) == 3
