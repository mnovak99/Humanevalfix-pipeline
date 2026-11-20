from pathlib import Path

import pandas as pd
import pytest

from pipeline.config import load_config
from pipeline.s01_prepare import enrich
from pipeline.s02_build_prompts import InvariantError, build_prompts, check_invariant
from pipeline.translations import TranslationError, load_and_validate, write_workbook

CFG = load_config()

ROWS = [
    {
        "task_id": "Python/1",
        "entry_point": "double",
        "bug_type": "value misuse",
        "prompt": 'def double(x):\n    """Return twice x.\n    >>> double(2)\n    4\n    """\n',
        "buggy_solution": "    return x + 2\n",
        "canonical_solution": "    return x * 2\n",
        "test": "def check(f):\n    assert f(2) == 4\n\ncheck(double)",
    },
    {
        "task_id": "Python/2",
        "entry_point": "fly",
        "bug_type": "missing logic",
        "prompt": "def fly(q):\n    '''\n    Return True if q flies.\n\n    Example:\n    fly([1]) ➞ True\n    For q = [2] the output should be False.\n    '''\n",
        "buggy_solution": "    return True\n",
        "canonical_solution": "    return q == [1]\n",
        "test": "def check(f):\n    assert f([1])\n\ncheck(fly)",
    },
]
TASKS = [enrich(r) for r in ROWS]
IDS = {t["task_id"] for t in TASKS}


def _mock(tmp_path: Path) -> Path:
    path = tmp_path / "translations.xlsx"
    write_workbook(path, TASKS, CFG, mock=True)
    return path


def _edit(path: Path, row_id: str, /, **changes: str) -> None:
    sheets = pd.read_excel(path, sheet_name=None, dtype=str, keep_default_na=False)
    df = sheets["docstrings"]
    for col, val in changes.items():
        df.loc[df.task_id == row_id, col] = val
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, d in sheets.items():
            d.to_excel(xw, sheet_name=name, index=False)


def _real_cs(path: Path) -> None:
    _edit(path, "Python/1", docstring_cs="Vrátí dvojnásobek x.\n{EXAMPLE_1}")
    _edit(path, "Python/2", docstring_cs="Vrátí True, pokud q letí.\n\nPříklad:\n{EXAMPLE_1}\nPro q = [2] by výstup měl být False.")


def test_mock_is_valid_and_flagged(tmp_path: Path) -> None:
    tr = load_and_validate(_mock(tmp_path), CFG.languages, TASKS, IDS)
    assert tr.is_mock and tr.mock_languages == {"cs"}
    assert tr.docstrings["cs"]["Python/1"].startswith("[CS] ")
    assert tr.instructions["cs"] == "[CS] Fix bugs in {entry_point}."


def test_real_translation_builds_prompts(tmp_path: Path) -> None:
    path = _mock(tmp_path)
    _real_cs(path)
    tr = load_and_validate(path, CFG.languages, TASKS, IDS)
    tr.instructions["cs"] = "Oprav chyby ve funkci {entry_point}."
    tr.mock_languages.clear()
    records = build_prompts(TASKS, tr, CFG.languages)
    assert len(records) == 4
    by_id = {r.prompt_id: r for r in records}
    en, cs = by_id["Python/2::en"].prompt, by_id["Python/2::cs"].prompt
    assert en.startswith("Fix bugs in fly.\n\n```python\ndef fly(q):\n    '''\n    Return True")
    assert cs.startswith("Oprav chyby ve funkci fly.\n\n```python\ndef fly(q):\n    '''\n    Vrátí True")
    assert "    fly([1]) ➞ True\n" in cs and cs.endswith("    return True\n```")
    assert not any(r.mock_translation for r in records)


@pytest.mark.parametrize(
    ("task_id", "changes", "message"),
    [
        ("Python/1", {"status": "translated"}, "not 'reviewed'"),
        ("Python/1", {"docstring_cs": ""}, "empty docstring_cs"),
        ("Python/1", {"docstring_cs": "Bez příkladu."}, "bad example markers"),
        ("Python/1", {"docstring_cs": 'Text """ {EXAMPLE_1}'}, "bad example markers"),
        ("Python/2", {"docstring_cs": "Text '''\n{EXAMPLE_1}"}, "triple quotes"),
        ("Python/1", {"docstring_en": "Edited EN.\n{EXAMPLE_1}"}, "differs from data/tasks.json"),
        ("Python/1", {"task_id": "Python/99"}, "missing task_ids"),
    ],
)
def test_validation_errors(tmp_path: Path, task_id: str, changes: dict, message: str) -> None:
    path = _mock(tmp_path)
    _edit(path, task_id, **changes)
    with pytest.raises(TranslationError, match=message):
        load_and_validate(path, CFG.languages, TASKS, IDS)


def test_unknown_extra_rows_rejected_but_known_allowed(tmp_path: Path) -> None:
    path = _mock(tmp_path)
    pilot = TASKS[:1]
    load_and_validate(path, CFG.languages, pilot, IDS)  # Python/2 is a known (full-set) task
    with pytest.raises(TranslationError, match="unknown task_ids"):
        load_and_validate(path, CFG.languages, pilot, {"Python/1"})


def test_warnings_for_lost_values_and_length(tmp_path: Path) -> None:
    path = _mock(tmp_path)
    _real_cs(path)
    _edit(path, "Python/2", docstring_cs="Letí?\n{EXAMPLE_1}\nPro q = [3] je výstup False.")
    tr = load_and_validate(path, CFG.languages, TASKS, IDS)
    assert any("Python/2" in w and "[2]" in w for w in tr.warnings)
    assert any("Python/2" in w and "length ratio" in w for w in tr.warnings)


def test_instruction_placeholder_mismatch(tmp_path: Path) -> None:
    path = _mock(tmp_path)
    sheets = pd.read_excel(path, sheet_name=None, dtype=str, keep_default_na=False)
    sheets["instructions"].loc[sheets["instructions"].language == "cs", "instruction"] = "Oprav chyby."
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, d in sheets.items():
            d.to_excel(xw, sheet_name=name, index=False)
    with pytest.raises(TranslationError, match="placeholders"):
        load_and_validate(path, CFG.languages, TASKS, IDS)


def test_invariant_detects_code_change() -> None:
    task = TASKS[0]
    ok = task["buggy_code"]
    check_invariant(task, {"en": ok, "cs": ok.replace("Return twice x.", "Vrátí dvojnásobek x.")})
    with pytest.raises(InvariantError, match="outside the docstring"):
        check_invariant(task, {"en": ok, "cs": ok.replace("x + 2", "x * 2")})
    with pytest.raises(InvariantError, match="outside the docstring"):
        check_invariant(task, {"en": ok, "cs": ok.replace("def double(x):", "def double(x):  ")})
    with pytest.raises(InvariantError, match="example 1"):
        check_invariant(task, {"en": ok, "cs": ok.replace(">>> double(2)", ">>> double(3)")})
