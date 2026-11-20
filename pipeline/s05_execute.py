"""Run code against the dataset tests in a subprocess.

`execute_program` is the single executor, also used by s01's sanity check.
"""

from __future__ import annotations

import logging
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from pipeline.config import RunContext
from pipeline.io import write_text

log = logging.getLogger(__name__)

PASSED, FAILED, ERROR, TIMEOUT, SKIPPED = "passed", "failed", "error", "timeout", "skipped"
STDERR_TAIL_CHARS = 500

# Last line of a traceback: "ValueError: msg", "AssertionError", "pkg.mod.Err: msg".
_EXC_LINE_RE = re.compile(r"^([A-Za-z_][\w.]*)(?::|$)")


@dataclass(frozen=True)
class ExecResult:
    status: str
    exc_type: str | None
    returncode: int | None
    duration_s: float
    stderr_tail: str

    def to_dict(self) -> dict:
        return asdict(self)


def build_program(code: str, test: str, preamble: str = "") -> str:
    """preamble (imports/helpers) + code + dataset test (which ends with `check(entry_point)`)."""
    parts = [p for p in (preamble, code, test) if p]
    return "\n\n".join(p.rstrip("\n") for p in parts) + "\n"


def _exception_type(stderr: str) -> str | None:
    lines = stderr.strip().splitlines()
    if not lines or "Traceback" not in stderr:
        return None
    m = _EXC_LINE_RE.match(lines[-1])
    return m.group(1).rsplit(".", 1)[-1] if m else None


def _kill_tree(proc: subprocess.Popen[bytes]) -> None:
    """Kill the process and its children (subprocess.run's timeout only kills the direct child)."""
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
            check=False,
        )
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.kill()


def execute_program(program: str, timeout_s: float, tmp_dir: Path, keep_file: bool = False) -> ExecResult:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / f"prog_{uuid.uuid4().hex}.py"
    write_text(path, program)
    popen_kwargs: dict = {}
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    t0 = time.perf_counter()
    try:
        # -I: isolated (ignore env/user site, no cwd on sys.path); -X utf8: UTF-8 I/O on Windows.
        proc = subprocess.Popen(
            [sys.executable, "-I", "-X", "utf8", str(path)],
            cwd=tmp_dir,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **popen_kwargs,
        )
        try:
            _, err_b = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            _, err_b = proc.communicate()
            return ExecResult(TIMEOUT, None, None, time.perf_counter() - t0, _tail(err_b))
        duration = time.perf_counter() - t0
        stderr = err_b.decode("utf-8", errors="replace")
        if proc.returncode == 0:
            return ExecResult(PASSED, None, 0, duration, _tail(err_b))
        exc = _exception_type(stderr)
        status = FAILED if exc == "AssertionError" else ERROR
        return ExecResult(status, exc or "unknown", proc.returncode, duration, _tail(err_b))
    finally:
        if not keep_file:
            path.unlink(missing_ok=True)


def _tail(err_b: bytes | None) -> str:
    text = (err_b or b"").decode("utf-8", errors="replace")
    return text[-STDERR_TAIL_CHARS:]


def run(ctx: RunContext) -> None:
    raise NotImplementedError("s05 step wiring (extracted.jsonl -> executions.jsonl) comes in Phase 3")
