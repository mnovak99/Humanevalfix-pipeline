"""Validate translations and build one prompt per task and language (data/prompts.jsonl)."""

from __future__ import annotations

import logging
import string
from dataclasses import asdict, dataclass

from pipeline.config import Config, RunContext
from pipeline.io import read_json, write_jsonl
from pipeline.validation import ValidationReport, validate

log = logging.getLogger(__name__)

CODE_PLACEHOLDER = "{declaration}{buggy_solution}"


class InvariantError(ValueError):
    pass


class TranslationCheckError(ValueError):
    pass


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    task_id: str
    language: str
    prompt: str
    entry_point: str

    def to_dict(self) -> dict:
        return asdict(self)


def prompt_id(task_id: str, language: str) -> str:
    return f"{task_id}::{language}"


def template_fields(template: str) -> set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(template) if name}


def check_templates(cfg: Config, sample_task: dict) -> None:
    """Every template must contain the code part exactly once and only use dataset fields."""
    for lang in cfg.languages:
        template = cfg.prompt_templates[lang]
        if template.count(CODE_PLACEHOLDER) != 1:
            raise InvariantError(f"template {lang!r} must contain {CODE_PLACEHOLDER} exactly once")
        unknown = template_fields(template) - set(sample_task)
        if unknown:
            raise InvariantError(f"template {lang!r} uses fields not in the dataset: {sorted(unknown)}")


def render(template: str, task: dict) -> str:
    return template.format(**task)


def check_invariant(task: dict, prompts: dict[str, str]) -> None:
    """The code part must appear once, byte-identical, in every language's prompt."""
    code = task["declaration"] + task["buggy_solution"]
    for lang, text in prompts.items():
        if text.count(code) != 1:
            raise InvariantError(f"{task['task_id']} [{lang}]: code part not found exactly once in the prompt")


def build_prompts(tasks: list[dict], cfg: Config) -> list[PromptRecord]:
    if tasks:
        check_templates(cfg, tasks[0])
    records: list[PromptRecord] = []
    for task in tasks:
        prompts = {lang: render(cfg.prompt_templates[lang], task) for lang in cfg.languages}
        check_invariant(task, prompts)
        records += [
            PromptRecord(prompt_id(task["task_id"], lang), task["task_id"], lang, prompts[lang], task["entry_point"])
            for lang in cfg.languages
        ]
    return records


def log_report(report: ValidationReport) -> None:
    for line in report.summary().split("\n"):
        log.info("translation check: %s", line)
    for issue in report.warnings:
        log.warning("translation %s [%s] %s: %s", issue.task_id, issue.language, issue.check, issue.detail)
    for issue in report.errors:
        log.error("translation %s [%s] %s: %s", issue.task_id, issue.language, issue.check, issue.detail)


def run(ctx: RunContext) -> None:
    cfg = ctx.config
    if not ctx.tasks_path.exists():
        raise FileNotFoundError(f"{ctx.tasks_path} not found; run step 01 first")
    tasks = read_json(ctx.tasks_path)["tasks"]

    report = validate(tasks, cfg.text_fields)
    log_report(report)
    if not report.ok:
        failing = sorted({i.task_id for i in report.errors}, key=lambda t: int(t.split("/")[1]))
        raise TranslationCheckError(f"translation checks failed for {len(failing)} task(s): {failing}")

    records = build_prompts(tasks, cfg)
    write_jsonl(ctx.prompts_path, (r.to_dict() for r in records))
    log.info(
        "invariant OK for %d tasks; wrote %d prompts (%s) to %s",
        len(tasks),
        len(records),
        ", ".join(cfg.languages),
        ctx.prompts_path,
    )
