import pytest

from pipeline.s04_extract import (
    compare_to_references,
    normalized_function,
    EMPTY_RESPONSE,
    FUNCTION_MISSING,
    NO_CODE_BLOCK,
    OK,
    SYNTAX_ERROR,
    extract,
)

FIXED = "def add(a, b):\n    return a + b\n"


@pytest.mark.parametrize(
    ("text", "status", "source"),
    [
        (f"Fixed:\n```python\n{FIXED}```\nDone.", OK, "python_block"),
        (f"```py\n{FIXED}```", OK, "python_block"),
        (f"```\n{FIXED}```", OK, "any_block"),
        (f"```text\n{FIXED}```", OK, "any_block"),
        (FIXED, NO_CODE_BLOCK, "whole_text"),
        ("The bug is in line 2, change - to +.", SYNTAX_ERROR, "whole_text"),
        ("```python\ndef add(a, b:\n    return a + b\n```", SYNTAX_ERROR, "python_block"),
        ("```python\ndef plus(a, b):\n    return a + b\n```", FUNCTION_MISSING, "python_block"),
        ("", EMPTY_RESPONSE, "none"),
        ("   \n", EMPTY_RESPONSE, "none"),
        (None, EMPTY_RESPONSE, "none"),
        (f"```python\n{FIXED}", OK, "python_block"),  # truncated: closing fence missing
    ],
)
def test_extract_cases(text: str | None, status: str, source: str) -> None:
    ex = extract(text, "add")
    assert (ex.status, ex.source) == (status, source)


def test_last_python_block_wins() -> None:
    text = (
        "Original:\n```python\ndef add(a, b):\n    return a - b\n```\n"
        "Some shell:\n```bash\npytest\n```\n"
        f"Fixed:\n```python\n{FIXED}```\n"
    )
    ex = extract(text, "add")
    assert ex.status == OK and ex.code == FIXED and ex.n_blocks == 3


def test_later_doctest_block_does_not_hide_the_fix() -> None:
    text = f"Fixed:\n```python\n{FIXED}```\nUsage:\n```python\n>>> add(1, 2)\n3\n```"
    ex = extract(text, "add")
    assert (ex.status, ex.source, ex.code) == (OK, "python_block", FIXED)


def test_later_block_without_the_function_does_not_hide_the_fix() -> None:
    text = f"```python\n{FIXED}```\nTry it:\n```python\nprint(add(1, 2))\n```"
    ex = extract(text, "add")
    assert ex.status == OK and ex.code == FIXED


def test_last_block_defining_the_function_wins() -> None:
    other = "def add(a, b):\n    return b + a\n"
    text = f"```python\n{FIXED}```\n```python\n{other}```\n```python\n>>> add(1, 2)\n```"
    assert extract(text, "add").code == other


def test_no_block_defines_function_reports_last_block() -> None:
    bad = "```python\ndef add(a, b:\n    return a\n```\n```python\n>>> add(1, 2)\n```"
    ex = extract(bad, "add")
    assert ex.status == SYNTAX_ERROR and ex.code.startswith(">>>")


def test_python_block_preferred_over_later_untagged_block() -> None:
    text = f"```python\n{FIXED}```\nOutput:\n```\n3\n```"
    ex = extract(text, "add")
    assert ex.status == OK and ex.code == FIXED


def test_keeps_imports_and_helpers() -> None:
    code = "import math\n\ndef helper(x):\n    return x\n\ndef add(a, b):\n    return helper(a) + b\n"
    ex = extract(f"```python\n{code}```", "add")
    assert ex.status == OK and ex.code == code


def test_indented_fence_in_markdown_list() -> None:
    ex = extract(f"1. Fix:\n   ```python\n{FIXED}   ```\n", "add")
    assert ex.status == OK


# --- comparison with reference solutions (exact_match_canonical / unchanged_from_buggy)


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        # canonical fix with a docstring, a comment and different formatting
        ('def add(x: int, y: int) -> int:\n    """Add."""\n    # fixed\n    return (x +   y)\n', (True, False)),
        ("def add(x: int, y: int) -> int:\n    return x - y\n", (False, True)),  # bug left in place
        ("def add(x: int, y: int) -> int:\n    return y + x\n", (False, False)),  # different but valid fix
        ("import math\n\ndef add(x: int, y: int) -> int:\n    return x + y\n", (True, False)),  # extra imports ok
        ("def plus(x, y):\n    return x + y\n", (None, None)),  # entry point missing
        (None, (None, None)),  # no usable code
    ],
)
def test_compare_to_references(task: dict, code: str | None, expected: tuple) -> None:
    assert compare_to_references(code, task) == expected


def test_normalized_function_strips_nested_docstrings() -> None:
    a = 'def f():\n    """Outer."""\n    def g():\n        """Inner."""\n        return 1\n    return g()\n'
    b = "def f():\n    def g():\n        return 1\n    return g()\n"
    assert normalized_function(a, "f") == normalized_function(b, "f")
