import pytest

from pipeline.docstrings import (
    DocstringError,
    DocstringParts,
    rebuild_source,
    split_docstring,
    strip_docstring,
)

DOCTEST = '''from typing import List


def has_close(numbers: List[float], t: float) -> bool:
    """ Check if any two numbers are closer
    than the threshold.
    >>> has_close([1.0, 2.0], 0.5)
    False
    >>> has_close([1.0, 2.8, 2.9], 0.3)
    True
    """
    return False
'''

ARROWS = """def will_it_fly(q, w):
    '''
    Write a function that returns True if q will fly.

    Example:
    will_it_fly([1, 2], 5) ➞ False
    # 1+2 is less than w, but it's unbalanced.

    will_it_fly([3], 5) ➞ True
    '''
    return True
"""

MULTILINE_CALL = '''def get_row(lst, x):
    """
    Find x in a nested list.

    Examples:
    get_row([
      [1,2,3],
      [1,2,1]
    ], 1) == [(0, 0), (1, 2), (1, 0)]
    get_row([], 1) == []
    """
    return []
'''

NO_EXAMPLES = '''def f(n):
    """Given n, return the count.
    Note: n >= 1. Input: n = 5
    Output: 1"""
    return n
'''

IMPORT_BEFORE = '''def max_fill(grid, capacity):
    import math
    """
    Fill buckets.
    max_fill([[1]], 1) == 1
    """
    return 1
'''

ALL = [DOCTEST, ARROWS, MULTILINE_CALL, NO_EXAMPLES, IMPORT_BEFORE]
NAMES = {DOCTEST: "has_close", ARROWS: "will_it_fly", MULTILINE_CALL: "get_row", NO_EXAMPLES: "f", IMPORT_BEFORE: "max_fill"}


@pytest.mark.parametrize("src", ALL)
def test_round_trip(src: str) -> None:
    p = split_docstring(src, NAMES[src])
    assert rebuild_source(src, p, p.description) == src
    assert DocstringParts.from_dict(p.to_dict()) == p


def test_doctest_examples_are_verbatim() -> None:
    p = split_docstring(DOCTEST, "has_close")
    assert p.description == " Check if any two numbers are closer\nthan the threshold.\n{EXAMPLE_1}"
    assert p.examples == [
        "    >>> has_close([1.0, 2.0], 0.5)\n    False\n    >>> has_close([1.0, 2.8, 2.9], 0.3)\n    True"
    ]


def test_arrow_examples_and_comments() -> None:
    p = split_docstring(ARROWS, "will_it_fly")
    assert len(p.examples) == 2
    # Comments between examples are prose and get translated.
    assert "# 1+2 is less than w, but it's unbalanced." in p.description.split("\n")


def test_multiline_call_is_one_block() -> None:
    p = split_docstring(MULTILINE_CALL, "get_row")
    assert len(p.examples) == 1
    assert p.examples[0].count("\n") == 4


def test_no_examples() -> None:
    p = split_docstring(NO_EXAMPLES, "f")
    assert p.examples == []
    assert p.description.endswith("Output: 1")


def test_translation_keeps_code_and_examples_identical() -> None:
    p = split_docstring(ARROWS, "will_it_fly")
    cs = "Napište funkci, která vrátí True, pokud q poletí.\n\nPříklad:\n{EXAMPLE_1}\n# 1+2 je méně než w.\n\n{EXAMPLE_2}"
    out = rebuild_source(ARROWS, p, cs)
    assert strip_docstring(out, split_docstring(out, "will_it_fly")) == strip_docstring(ARROWS, p)
    for block in p.examples:
        assert block in out
    assert "    Napište funkci" in out


@pytest.mark.parametrize(
    "bad",
    [
        "Text bez příkladů.",  # markers dropped
        "{EXAMPLE_2}\n{EXAMPLE_1}",  # reordered
        "Viz {EXAMPLE_1} a {EXAMPLE_2}",  # not on own lines
    ],
)
def test_bad_markers_rejected(bad: str) -> None:
    p = split_docstring(ARROWS, "will_it_fly")
    with pytest.raises(DocstringError):
        rebuild_source(ARROWS, p, bad)


def test_missing_docstring() -> None:
    with pytest.raises(DocstringError):
        split_docstring("def g():\n    return 1\n", "g")
