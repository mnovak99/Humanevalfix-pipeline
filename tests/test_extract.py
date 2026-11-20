import pytest

from pipeline.s04_extract import (
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
