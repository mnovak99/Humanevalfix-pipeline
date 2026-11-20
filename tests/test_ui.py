import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from pipeline.config import load_config
from pipeline.ui.data import missing_api_keys
from pipeline.ui.runner import Runner, pipeline_args, read_log_tail

APP = str(Path(__file__).resolve().parent.parent / "pipeline" / "ui" / "app.py")
SLEEPER = [sys.executable, "-c", "import time; print('started', flush=True); time.sleep(60)"]


@pytest.mark.parametrize(
    ("command", "kwargs", "expected"),
    [
        ("all", {"pilot": True, "dry_run": True}, ["run", "--all", "--pilot", "--dry-run"]),
        ("step", {"step": "03", "yes": True}, ["run", "--step", "03", "--yes"]),
        ("smoke", {"n_tasks": 3, "pilot": True, "yes": True}, ["smoke", "--n-tasks", "3", "--yes"]),
        ("smoke", {"dry_run": True}, ["smoke", "--n-tasks", "2", "--dry-run"]),
    ],
)
def test_pipeline_args(command: str, kwargs: dict, expected: list[str]) -> None:
    assert pipeline_args(command, **kwargs) == expected


def test_runner_runs_pipeline_and_logs(tmp_path: Path) -> None:
    runner = Runner(log_dir=tmp_path)
    run = runner.start(["--help"])
    run.proc.wait(timeout=60)
    assert run.state == "finished"
    assert "usage: python -m pipeline" in read_log_tail(run.log_path)


def test_runner_busy_guard_and_stop(tmp_path: Path) -> None:
    runner = Runner(log_dir=tmp_path, base_cmd=SLEEPER)
    run = runner.start([])
    try:
        assert runner.busy()
        with pytest.raises(RuntimeError, match="already in progress"):
            runner.start([])
    finally:
        runner.stop()
    assert run.state == "stopped" and not runner.busy()


def test_read_log_tail_handles_progress_bars(tmp_path: Path) -> None:
    log = tmp_path / "x.log"
    log.write_text("start\ncalls: 10%\rcalls: 50%\rcalls: 100%\ndone\n", encoding="utf-8")
    assert read_log_tail(log) == "start\ncalls: 100%\ndone"
    assert read_log_tail(log, n=1) == "done"


def test_missing_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert missing_api_keys(load_config()) == ["OPENAI_API_KEY"]


def test_app_renders_without_errors() -> None:
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    assert [t.label for t in at.tabs] == ["Overview", "Run", "Translations", "Results"]


def test_paid_run_needs_confirmation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.toggle[0].set_value(False).run()  # live mode
    next(r for r in at.radio if r.label == "What to run").set_value("smoke").run()
    assert not at.exception, at.exception
    start = next(b for b in at.button if b.label == "Start")
    assert start.disabled, "a paid run must not start before the confirmation box is ticked"
    confirm = next(c for c in at.checkbox if c.label.startswith("I confirm a paid run"))
    confirm.check().run()
    assert not next(b for b in at.button if b.label == "Start").disabled
