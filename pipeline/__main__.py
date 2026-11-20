"""CLI entry point: python -m pipeline ..."""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from collections.abc import Callable

from pipeline.config import RunContext, load_config, load_env
from pipeline.docstrings import DocstringError
from pipeline.s02_build_prompts import InvariantError
from pipeline.s03_run_models import LiveRunRefused
from pipeline.translations import TranslationError

# Expected, user-facing failures: report the message, no traceback.
PipelineError = (TranslationError, DocstringError, InvariantError, LiveRunRefused, FileNotFoundError)

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

    mock = sub.add_parser("mock-translations", help="create a mock data/translations.xlsx ([CS] + EN)")
    mock.add_argument("--force", action="store_true", help="overwrite a file that contains real translations")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpx2", "huggingface_hub", "datasets", "filelock", "fsspec"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    load_env()
    config = load_config()

    try:
        if args.command == "mock-translations":
            from pipeline import translations

            translations.write_mock(RunContext(config=config), force=args.force)
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
