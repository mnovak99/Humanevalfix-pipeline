import pytest


@pytest.fixture
def task() -> dict:
    """A small task in the shape of data/humanevalpack_cs.jsonl."""
    return {
        "task_id": "Python/1",
        "entry_point": "add",
        "bug_type": "operator misuse",
        "declaration": "def add(x: int, y: int) -> int:\n",
        "buggy_solution": "    return x - y\n",
        "canonical_solution": "    return x + y\n",
        "test": "def check(add):\n    assert add(2, 3) == 5\n\ncheck(add)",
        "instruction": (
            "Write a Python function `add(x: int, y: int) -> int` to solve the following problem:\n"
            "Add two numbers x and y\n>>> add(2, 3)\n5\n>>> add(5, 7)\n12"
        ),
        "docstring_translated": (
            "Napiš funkci v Pythonu `add(x: int, y: int) -> int`, která řeší následující problém:\n"
            "Sečti dvě čísla x a y\n>>> add(2, 3)\n5\n>>> add(5, 7)\n12"
        ),
    }
