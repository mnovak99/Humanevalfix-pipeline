"""Start, watch and stop one pipeline run as a background process.

Runs are `python -m pipeline ...` subprocesses with output in a log file, so a long run
keeps going when the browser tab is closed or the page reloads.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from pipeline.config import ROOT
from pipeline.s05_execute import kill_tree

LOG_DIR = ROOT / "outputs" / "tmp" / "ui_runs"
TAIL_LINES = 200


@dataclass
class Run:
    args: list[str]
    log_path: Path
    started: datetime
    proc: subprocess.Popen = field(repr=False)
    stopped_by_user: bool = False

    @property
    def returncode(self) -> int | None:
        return self.proc.poll()

    @property
    def running(self) -> bool:
        return self.returncode is None

    @property
    def label(self) -> str:
        return " ".join(self.args)

    @property
    def state(self) -> str:
        if self.running:
            return "running"
        if self.stopped_by_user:
            return "stopped"
        return "finished" if self.returncode == 0 else f"failed (exit {self.returncode})"


def read_log_tail(path: Path, n: int = TAIL_LINES) -> str:
    """Last n lines; progress bars rewrite their line with \\r, so keep the last segment."""
    if not path.exists():
        return ""
    # newline="" keeps lone \r (universal newlines would turn each progress update into a line).
    with path.open(encoding="utf-8", errors="replace", newline="") as f:
        text = f.read().replace("\r\n", "\n")
    lines = [ln.rsplit("\r", 1)[-1] for ln in text.split("\n")]
    return "\n".join([ln for ln in lines if ln.strip()][-n:])


class Runner:
    """At most one run at a time (runs write to shared files)."""

    def __init__(self, log_dir: Path = LOG_DIR, cwd: Path = ROOT, base_cmd: list[str] | None = None) -> None:
        self._lock = threading.Lock()
        self._log_dir = log_dir
        self._cwd = cwd
        self._base_cmd = base_cmd or [sys.executable, "-m", "pipeline"]
        self.current: Run | None = None

    def busy(self) -> bool:
        return self.current is not None and self.current.running

    def start(self, args: list[str]) -> Run:
        with self._lock:
            if self.busy():
                raise RuntimeError(f"a run is already in progress: {self.current.label}")  # type: ignore[union-attr]
            self._log_dir.mkdir(parents=True, exist_ok=True)
            started = datetime.now()
            log_path = self._log_dir / f"{started:%Y%m%d-%H%M%S}.log"
            env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
            kwargs: dict = {}
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
            else:
                kwargs["start_new_session"] = True
            with log_path.open("w", encoding="utf-8") as log:
                log.write(f"$ python -m pipeline {' '.join(args)}\n")
                log.flush()
                proc = subprocess.Popen(
                    [*self._base_cmd, *args],
                    cwd=self._cwd,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    **kwargs,
                )
            self.current = Run(args, log_path, started, proc)
            return self.current

    def stop(self) -> None:
        with self._lock:
            if self.busy():
                assert self.current is not None
                self.current.stopped_by_user = True
                kill_tree(self.current.proc)
                self.current.proc.wait(timeout=10)


def pipeline_args(
    command: str, *, pilot: bool = False, dry_run: bool = False, yes: bool = False, step: str | None = None, n_tasks: int | None = None
) -> list[str]:
    """CLI arguments for a UI action. `yes` is passed only after the user confirmed a paid run."""
    if command == "smoke":
        args = ["smoke", "--n-tasks", str(n_tasks or 2)]
    elif command == "all":
        args = ["run", "--all"]
    elif command == "step":
        assert step is not None
        args = ["run", "--step", step]
    elif command == "mock-translations":
        return ["mock-translations"]
    else:
        raise ValueError(f"unknown command {command!r}")
    if pilot and command != "smoke":
        args.append("--pilot")
    if dry_run:
        args.append("--dry-run")
    if yes:
        args.append("--yes")
    return args
