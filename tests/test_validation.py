from pipeline.validation import (
    BACKTICKS,
    DOCTEST,
    ENTRY_POINT,
    LENGTH,
    NUMBERS,
    doctest_pairs,
    numbers,
    validate,
)

FIELDS = {"en": "instruction", "cs": "docstring_translated"}


def _checks(task: dict) -> tuple[set[str], set[str]]:
    report = validate([task], FIELDS)
    return {i.check for i in report.errors}, {i.check for i in report.warnings}


def test_valid_translation_passes(task: dict) -> None:
    report = validate([task], FIELDS)
    assert report.ok and not report.warnings


def test_doctest_pairs_include_expected_output() -> None:
    text = "intro\n  >>> f(1)\n  2\n>>> f(2)\n>>> f(3)\n4"
    assert doctest_pairs(text) == [(">>> f(1)", "2"), (">>> f(2)", ">>> f(3)"), (">>> f(3)", "4")]


def test_changed_doctest_call_is_error(task: dict) -> None:
    task["docstring_translated"] = task["docstring_translated"].replace(">>> add(2, 3)", ">>> přidat(2, 3)")
    assert DOCTEST in _checks(task)[0]


def test_changed_expected_output_is_error(task: dict) -> None:
    task["docstring_translated"] = task["docstring_translated"].replace("\n12", "\n13")
    assert DOCTEST in _checks(task)[0]


def test_decimal_comma_in_doctest_is_error(task: dict) -> None:
    task["instruction"] += "\n>>> add(0.5, 1)\n1.5"
    task["docstring_translated"] += "\n>>> add(0,5, 1)\n1,5"
    assert DOCTEST in _checks(task)[0]


def test_missing_doctest_is_error(task: dict) -> None:
    task["docstring_translated"] = task["docstring_translated"].split("\n>>> add(5, 7)")[0]
    assert DOCTEST in _checks(task)[0]


def test_entry_point_missing_is_error(task: dict) -> None:
    task["instruction"] = "Write a function that adds two numbers."
    task["docstring_translated"] = "Napiš funkci, která sečte dvě čísla."
    errors, _ = _checks(task)
    assert errors == {ENTRY_POINT}


def test_backticked_identifier_missing_is_error(task: dict) -> None:
    task["docstring_translated"] = task["docstring_translated"].replace("`add(x: int, y: int) -> int`", "`add(x, y)`")
    assert BACKTICKS in _checks(task)[0]


def test_length_ratio_is_warning_only(task: dict) -> None:
    task["docstring_translated"] += "\n" + "Velmi dlouhé vysvětlení. " * 30
    errors, warnings = _checks(task)
    assert not errors and LENGTH in warnings


def test_missing_number_is_warning_only(task: dict) -> None:
    task["instruction"] += "\nThe result is at most 100."
    task["docstring_translated"] += "\nVýsledek je nejvýše sto."
    errors, warnings = _checks(task)
    assert not errors and NUMBERS in warnings


def test_numbers_ignore_identifiers_and_split_decimal_commas() -> None:
    assert numbers("x2 = 3.5, f1(10) -0.5") == {"3.5", "10", "-0.5"}
    assert "1.5" not in numbers("1,5")


def test_report_lists_offending_ids(task: dict) -> None:
    bad = {**task, "task_id": "Python/7", "docstring_translated": task["docstring_translated"].replace("\n5\n", "\n6\n")}
    report = validate([task, bad], FIELDS)
    assert report.ids(DOCTEST) == ["Python/7"]
    assert "Python/7" in report.summary()
