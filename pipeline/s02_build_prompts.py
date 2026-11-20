"""Validate translations and build per-language prompts (data/prompts.jsonl)."""

from __future__ import annotations

import ast
import logging
from dataclasses import asdict, dataclass

from pipeline.config import RunContext
from pipeline.docstrings import DocstringParts, locate_docstring, rebuild_source
from pipeline.io import read_json, write_jsonl
from pipeline.translations import Translations, load_and_validate

log = logging.getLogger(__name__)

PROMPT_TEMPLATE = "{instruction}\n\n```python\n{code}\n```"


class InvariantError(ValueError):
    pass


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    task_id: str
    language: str
    prompt: str
    entry_point: str
    mock_translation: bool

    def to_dict(self) -> dict:
        return asdict(self)


def prompt_id(task_id: str, language: str) -> str:
    return f"{task_id}::{language}"


def code_without_docstring(code: str, entry_point: str) -> str:
    start, end = locate_docstring(code, entry_point)
    return code[:start] + code[end:]


def build_code(task: dict, description: str) -> str:
    parts = DocstringParts.from_dict(task["docstring_parts"])
    code = rebuild_source(task["buggy_code"], parts, description)
    try:
        ast.parse(code)
    except SyntaxError as exc:
        raise InvariantError(f"{task['task_id']}: code with translated docstring does not parse: {exc}") from exc
    return code


def check_invariant(task: dict, codes: dict[str, str]) -> None:
    """Code minus docstring content must be byte-identical across languages; examples verbatim."""
    entry = task["entry_point"]
    reference = code_without_docstring(task["buggy_code"], entry)
    examples = task["docstring_parts"]["examples"]
    for lang, code in codes.items():
        if code_without_docstring(code, entry) != reference:
            raise InvariantError(f"{task['task_id']} [{lang}]: code outside the docstring differs from the original")
        start, end = locate_docstring(code, entry)
        content, pos = code[start:end], 0
        for k, block in enumerate(examples, 1):
            found = content.find(block, pos)
            if found < 0:
                raise InvariantError(f"{task['task_id']} [{lang}]: example {k} is not verbatim in the docstring")
            pos = found + len(block)
    if "en" in codes and codes["en"] != task["buggy_code"]:
        raise InvariantError(f"{task['task_id']} [en]: rebuilt English code differs from the original")


def build_prompt(instruction_template: str, task: dict, code: str) -> str:
    instruction = instruction_template.format(entry_point=task["entry_point"])
    return PROMPT_TEMPLATE.format(instruction=instruction, code=code.rstrip("\n"))


def build_prompts(tasks: list[dict], tr: Translations, languages: list[str]) -> list[PromptRecord]:
    records: list[PromptRecord] = []
    for task in tasks:
        codes = {lang: build_code(task, tr.docstrings[lang][task["task_id"]]) for lang in languages}
        check_invariant(task, codes)
        for lang in languages:
            records.append(
                PromptRecord(
                    prompt_id=prompt_id(task["task_id"], lang),
                    task_id=task["task_id"],
                    language=lang,
                    prompt=build_prompt(tr.instructions[lang], task, codes[lang]),
                    entry_point=task["entry_point"],
                    mock_translation=lang in tr.mock_languages,
                )
            )
    _check_template_invariant(records, languages)
    return records


def _check_template_invariant(records: list[PromptRecord], languages: list[str]) -> None:
    """Same number of prompts per language and per task (guards against silent drops)."""
    per_task: dict[str, set[str]] = {}
    for r in records:
        per_task.setdefault(r.task_id, set()).add(r.language)
    bad = [tid for tid, langs in per_task.items() if langs != set(languages)]
    if bad:
        raise InvariantError(f"tasks without a prompt in every language: {bad}")


def run(ctx: RunContext) -> None:
    cfg = ctx.config
    if not ctx.tasks_path.exists():
        raise FileNotFoundError(f"{ctx.tasks_path} not found; run step 01 first")
    tasks = read_json(ctx.tasks_path)["tasks"]
    full_path = RunContext(cfg, root=ctx.root).tasks_path
    known = {t["task_id"] for t in read_json(full_path)["tasks"]} if full_path.exists() else set()

    tr = load_and_validate(cfg.translations_path, cfg.languages, tasks, known)
    for w in tr.warnings:
        log.warning("translation: %s", w)
    if tr.is_mock:
        log.warning("MOCK translations in use for %s: dry runs only", sorted(tr.mock_languages))

    records = build_prompts(tasks, tr, cfg.languages)
    write_jsonl(ctx.prompts_path, (r.to_dict() for r in records))
    log.info(
        "invariant OK for %d tasks; wrote %d prompts (%s) to %s",
        len(tasks),
        len(records),
        ", ".join(cfg.languages),
        ctx.prompts_path,
    )
