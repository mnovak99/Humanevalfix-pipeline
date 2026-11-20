import time
from pathlib import Path

import pytest

from pipeline.s05_execute import ERROR, FAILED, PASSED, TIMEOUT, build_program, execute_program

TEST = "def check(f):\n    assert f(2) == 4\n\ncheck(double)"


def _run(code: str, tmp_path: Path, timeout: float = 5) -> object:
    return execute_program(build_program(code, TEST), timeout, tmp_path)


def test_passed(tmp_path: Path) -> None:
    r = _run("def double(x):\n    return x * 2\n", tmp_path)
    assert r.status == PASSED and r.exc_type is None


def test_failed_on_assertion(tmp_path: Path) -> None:
    r = _run("def double(x):\n    return x + 1\n", tmp_path)
    assert r.status == FAILED and r.exc_type == "AssertionError"


def test_error_stores_exception_type(tmp_path: Path) -> None:
    r = _run("def double(x):\n    return x / 0\n", tmp_path)
    assert r.status == ERROR and r.exc_type == "ZeroDivisionError"
    assert "ZeroDivisionError" in r.stderr_tail and len(r.stderr_tail) <= 500


def test_error_on_missing_function(tmp_path: Path) -> None:
    r = _run("def triple(x):\n    return x * 3\n", tmp_path)
    assert r.status == ERROR and r.exc_type == "NameError"


def test_timeout_kills_process(tmp_path: Path) -> None:
    r = _run("def double(x):\n    while True:\n        pass\n", tmp_path, timeout=1)
    assert r.status == TIMEOUT
    assert r.duration_s < 5


def test_timeout_kills_child_processes(tmp_path: Path) -> None:
    marker = tmp_path / "child_alive.txt"
    child = tmp_path / "child.py"
    child.write_text(
        f"import pathlib, time\ntime.sleep(3)\npathlib.Path({str(marker)!r}).write_text('x')\n", encoding="utf-8"
    )
    code = (
        "import subprocess, sys\n"
        "def double(x):\n"
        f"    subprocess.Popen([sys.executable, {str(child)!r}])\n"
        "    while True:\n"
        "        pass\n"
    )
    r = _run(code, tmp_path / "run", timeout=1)
    assert r.status == TIMEOUT, r.stderr_tail
    time.sleep(4)
    assert not marker.exists(), "child process survived the timeout"


def test_temp_file_removed(tmp_path: Path) -> None:
    _run("def double(x):\n    return 2 * x\n", tmp_path)
    assert list(tmp_path.glob("prog_*.py")) == []


@pytest.mark.parametrize("text", ["čeština ➞ ≤", "naïve"])
def test_unicode_output(tmp_path: Path, text: str) -> None:
    r = _run(f"def double(x):\n    print({text!r})\n    return x * 2\n", tmp_path)
    assert r.status == PASSED
