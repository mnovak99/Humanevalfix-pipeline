"""CLI entry point: python -m pipeline ..."""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from collections.abc import Callable

from pipeline.config import MissingApiKey, RunContext, load_config, load_env
from pipeline.dataset import DatasetError
from pipeline.s02_build_prompts import InvariantError, TranslationCheckError
from pipeline.s03_run_models import LiveRunRefused

# Expected, user-facing failures: report the message, no traceback.
PipelineError = (
    DatasetError,
    TranslationCheckError,
    InvariantError,
    LiveRunRefused,
    MissingApiKey,
    FileNotFoundError,
)

STEPS: dict[str, str] = {
    "01": "pipeline.s01_prepare",
    "02": "pipeline.s02_build_prompts",
    "03": "pipeline.s03_run_models",
    "04": "pipeline.s04_extract",
    "05": "pipeline.s05_execute",
    "06": "pipeline.s06_aggregate",
}

log = logging.getLogger("pipeline")


def _step_fn(step: str) -> Callable[[RunContext], None]:
    return importlib.import_module(STEPS[step]).run


def run_ui(port: int, open_browser: bool = True) -> int:
    import subprocess
    import threading
    import webbrowser
    from pathlib import Path

    app = Path(__file__).resolve().parent / "ui" / "app.py"
    url = f"http://127.0.0.1:{port}"
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app),
        "--server.address", "127.0.0.1",  # local only: the UI can start paid runs
        "--server.port", str(port),
        "--server.headless", "true",  # skips Streamlit's first-run email prompt
        "--browser.gatherUsageStats", "false",
        "--client.toolbarMode", "minimal",  # hides the "Deploy" button
    ]  # fmt: skip
    log.info("web UI at %s (close this window or press Ctrl+C to stop)", url)
    if open_browser:
        threading.Timer(3.0, webbrowser.open, args=(url,)).start()
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m pipeline")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run one pipeline step or all of them")
    which = run.add_mutually_exclusive_group(required=True)
    which.add_argument("--step", choices=sorted(STEPS), help="step number, e.g. 01")
    which.add_argument("--all", action="store_true", help="run steps 01-06 in order")
    run.add_argument("--pilot", action="store_true", help="use pilot_n_tasks; writes *_pilot files")
    run.add_argument("--dry-run", action="store_true", help="fake provider, no API calls; writes *_dry files")
    run.add_argument("--yes", action="store_true", help="skip the cost confirmation for live runs")

    smoke = sub.add_parser("smoke", help="EN-only check of the real APIs on a few tasks (writes *_smoke files)")
    smoke.add_argument("--n-tasks", type=int, default=2, help="number of tasks (default 2)")
    smoke.add_argument("--dry-run", action="store_true", help="fake provider, no API calls")
    smoke.add_argument("--yes", action="store_true", help="skip the cost confirmation")

    ui = sub.add_parser("ui", help="open the web UI (local only, http://127.0.0.1:8501)")
    ui.add_argument("--port", type=int, default=8501)
    ui.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpx2"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    load_env()
    config = load_config()

    try:
        if args.command == "ui":
            return run_ui(args.port, open_browser=not args.no_browser)
        if args.command == "smoke":
            from pipeline.smoke import run_smoke

            run_smoke(config, args.n_tasks, yes=args.yes, dry_run=args.dry_run)
            return 0

        ctx = RunContext(config=config, pilot=args.pilot, dry_run=args.dry_run, yes=args.yes)
        steps = sorted(STEPS) if args.all else [args.step]
        for step in steps:
            log.info("=== step %s (pilot=%s, dry_run=%s) ===", step, ctx.pilot, ctx.dry_run)
            _step_fn(step)(ctx)
    except NotImplementedError as exc:
        log.error("%s", exc)
        return 2
    except PipelineError as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
