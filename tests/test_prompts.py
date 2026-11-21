from dataclasses import replace

import pytest

from pipeline.config import load_config
from pipeline.s01_prepare import enrich
from pipeline.s02_build_prompts import InvariantError, build_prompts, check_invariant, check_templates
from pipeline.smoke import build_smoke_prompts

CFG = load_config()

EXPECTED_EN = (
    "I gave an AI this prompt, but the result doesn't really work. Could you please fix it?\n"
    "Used prompt: Write a Python function `add(x: int, y: int) -> int` to solve the following problem:\n"
    "Add two numbers x and y\n>>> add(2, 3)\n5\n>>> add(5, 7)\n12\n"
    "Code: def add(x: int, y: int) -> int:\n    return x - y\n\n"
    "Return the complete corrected function in a single Python code block."
)
EXPECTED_CS = (
    "Dal jsem AI toto zadání, ale výsledek nějak nefunguje. Můžeš to prosím opravit?\n"
    "Použité zadání: Napiš funkci v Pythonu `add(x: int, y: int) -> int`, která řeší následující problém:\n"
    "Sečti dvě čísla x a y\n>>> add(2, 3)\n5\n>>> add(5, 7)\n12\n"
    "Kód: def add(x: int, y: int) -> int:\n    return x - y\n\n"
    "Vrať celou opravenou funkci v jednom bloku kódu v Pythonu."
)


def test_prompts_match_the_templates_exactly(task: dict) -> None:
    records = {r.language: r for r in build_prompts([enrich(task)], CFG)}
    assert records["en"].prompt == EXPECTED_EN
    assert records["cs"].prompt == EXPECTED_CS
    assert records["cs"].prompt_id == "Python/1::cs" and records["cs"].entry_point == "add"


def test_code_part_is_identical_across_languages(task: dict) -> None:
    records = build_prompts([enrich(task)], CFG)
    code = task["declaration"] + task["buggy_solution"]
    assert all(r.prompt.count(code) == 1 for r in records)


def test_invariant_fails_when_code_differs(task: dict) -> None:
    good = {"en": "x " + task["declaration"] + task["buggy_solution"]}
    check_invariant(task, good)
    with pytest.raises(InvariantError):
        check_invariant(task, {"en": good["en"], "cs": "Kód: " + task["declaration"] + "    return x + y\n"})


@pytest.mark.parametrize(
    "template",
    [
        "Oprav: {docstring_translated}\nKód: {declaration}\n{buggy_solution}",  # code part split
        "Oprav: {docstring_translated}\n{declaration}{buggy_solution}\n{declaration}{buggy_solution}",  # twice
        "Oprav: {nonexistent}\n{declaration}{buggy_solution}",  # unknown field
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
