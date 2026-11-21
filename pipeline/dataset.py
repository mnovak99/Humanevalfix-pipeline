"""Load the task dataset (data/humanevalpack_cs.jsonl), the pipeline's only data source."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pipeline.io import read_text

# Fields the pipeline reads; the file may carry more (e.g. `text`, `docstring`).
REQUIRED_FIELDS = (
    "task_id",
    "declaration",
    "buggy_solution",
    "canonical_solution",
    "test",
    "entry_point",
    "bug_type",
)


class DatasetError(ValueError):
    pass


def load_rows(path: Path, text_fields: dict[str, str] | None = None) -> list[dict]:
    """Rows in file order; checks required fields, text fields and unique ids."""
    if not path.exists():
        raise DatasetError(f"{path} not found")
    rows = [json.loads(line) for line in read_text(path).splitlines() if line.strip()]
    required = [*REQUIRED_FIELDS, *(text_fields or {}).values()]
    problems = []
    for i, row in enumerate(rows, 1):
        missing = [f for f in required if not isinstance(row.get(f), str) or not row[f]]
        if missing:
            problems.append(f"line {i} ({row.get('task_id', '?')}): missing/empty {missing}")
    ids = [r.get("task_id") for r in rows]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        problems.append(f"duplicate task_ids: {dupes}")
    if problems:
        raise DatasetError(f"{path}: {len(problems)} problem(s):\n  - " + "\n  - ".join(problems[:20]))
    return rows


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
