from dataclasses import replace

import pytest

from pipeline.config import load_config
from pipeline.dataset import load_rows
from pipeline.s01_prepare import enrich
from pipeline.s02_build_prompts import InvariantError, build_prompts, check_invariant, check_templates, code_part
from pipeline.smoke import build_smoke_prompts

CFG = load_config()

EXPECTED_EN = """\
I gave an AI this prompt, but the result doesn't really work. Could you please fix it?

Used prompt:
Write a Python function `add(x: int, y: int) -> int` to solve the following problem:
Add two numbers x and y
>>> add(2, 3)
5
>>> add(5, 7)
12

Code:
def add(x: int, y: int) -> int:
    return x - y

Return the complete corrected function in a single Python code block."""

EXPECTED_CS = """\
Dal jsem AI toto zadání, ale výsledek nějak nefunguje. Můžeš to prosím opravit?

Použité zadání:
Napiš funkci v Pythonu `add(x: int, y: int) -> int`, která řeší následující problém:
Sečti dvě čísla x a y
>>> add(2, 3)
5
>>> add(5, 7)
12

Kód:
def add(x: int, y: int) -> int:
    return x - y

Vrať celou opravenou funkci v jednom bloku kódu v Pythonu."""

# Python/23 from data/humanevalpack_cs.jsonl, rendered with the templates in config.yaml.
EXPECTED_23_EN = """\
I gave an AI this prompt, but the result doesn't really work. Could you please fix it?

Used prompt:
Write a Python function `strlen(string: str) -> int` to solve the following problem:
Return length of given string
>>> strlen('')
0
>>> strlen('abc')
3

Code:
def strlen(string: str) -> int:
    return len(string) - 1

Return the complete corrected function in a single Python code block."""

EXPECTED_23_CS = """\
Dal jsem AI toto zadání, ale výsledek nějak nefunguje. Můžeš to prosím opravit?

Použité zadání:
Napiš v Pythonu funkci `strlen(string: str) -> int`, která řeší následující problém:
Vrať délku daného řetězce
>>> strlen('')
0
>>> strlen('abc')
3

Kód:
def strlen(string: str) -> int:
    return len(string) - 1

Vrať celou opravenou funkci v jednom bloku kódu v Pythonu."""


def test_prompts_match_the_templates_exactly(task: dict) -> None:
    records = {r.language: r for r in build_prompts([enrich(task)], CFG)}
    assert records["en"].prompt == EXPECTED_EN
    assert records["cs"].prompt == EXPECTED_CS
    assert records["cs"].prompt_id == "Python/1::cs" and records["cs"].entry_point == "add"


def test_python_23_prompts_from_dataset_exactly() -> None:
    row = next(r for r in load_rows(CFG.dataset_path, CFG.text_fields) if r["task_id"] == "Python/23")
    records = {r.language: r.prompt for r in build_prompts([enrich(row)], CFG)}
    assert records["en"] == EXPECTED_23_EN
    assert records["cs"] == EXPECTED_23_CS


def test_templates_keep_line_breaks_and_blank_lines() -> None:
    # Guards against a folded YAML block (>), which would join lines and drop blank lines.
    for lang, template in CFG.prompt_templates.items():
        assert "\n\n" in template, lang
        assert "\n{declaration}\n{buggy_solution}\n\n" in template, lang
        assert not template.endswith("\n"), lang


def test_no_double_blank_line_outside_the_code() -> None:
    """No "\\n\\n\\n" anywhere except inside the original code.

    25 declarations contain PEP 8 blank lines between imports and the function
    (e.g. Python/0: "from typing import List\\n\\n\\ndef ..."); that is task code and
    must not be changed, so it is the only place a triple newline may appear.
    """
    rows = {r["task_id"]: enrich(r) for r in load_rows(CFG.dataset_path, CFG.text_fields)}
    records = build_prompts(list(rows.values()), CFG)
    assert len(records) == 2 * 164
    offending = [r.prompt_id for r in records if "\n\n\n" in r.prompt.replace(code_part(rows[r.task_id]), "<code>")]
    assert offending == []


def test_only_trailing_newlines_are_stripped(task: dict) -> None:
    task["buggy_solution"] = "    return x - y \n\n"  # trailing space must stay
    task["docstring_translated"] += "\n\n"
    cs = next(r.prompt for r in build_prompts([enrich(task)], CFG) if r.language == "cs")
    assert "\n12\n\nKód:\ndef add(x: int, y: int) -> int:\n    return x - y \n\nVrať" in cs
    assert "\n\n\n" not in cs


def test_code_part_is_identical_across_languages(task: dict) -> None:
    records = build_prompts([enrich(task)], CFG)
    code = (task["declaration"] + task["buggy_solution"]).rstrip("\n")
    assert all(r.prompt.count(code) == 1 for r in records)


def test_code_part_requires_single_newline_after_declaration(task: dict) -> None:
    task["declaration"] += "\n"  # template line break would no longer reproduce the code
    with pytest.raises(InvariantError, match="exactly one newline"):
        build_prompts([enrich(task)], CFG)


def test_invariant_fails_when_code_differs(task: dict) -> None:
    good = {"en": "x " + task["declaration"] + task["buggy_solution"]}
    check_invariant(task, good)
    with pytest.raises(InvariantError):
        check_invariant(task, {"en": good["en"], "cs": "Kód: " + task["declaration"] + "    return x + y\n"})


@pytest.mark.parametrize(
    "template",
    [
        "Oprav: {docstring_translated}\nKód: {declaration}{buggy_solution}",  # no line break in code part
        "Oprav: {docstring_translated}\n{declaration}\n{buggy_solution}\n{declaration}\n{buggy_solution}",  # twice
        "Oprav: {nonexistent}\n{declaration}\n{buggy_solution}",  # unknown field
    ],
)
def test_bad_templates_rejected(task: dict, template: str) -> None:
    cfg = replace(CFG, prompt_templates={**CFG.prompt_templates, "cs": template})
    with pytest.raises(InvariantError):
        check_templates(cfg, task)


def test_smoke_prompts_are_the_english_prompts(task: dict) -> None:
    t = enrich(task)
    en = [r for r in build_prompts([t], CFG) if r.language == "en"]
    assert build_smoke_prompts([t], CFG) == en
