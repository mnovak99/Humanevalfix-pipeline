"""Read-only analysis of task Python/132 across the first pilot and the noreason pilot.

Run from the repo root:  uv run python analysis/python132/analyze132.py > analysis/python132/out132.txt
No API calls; nothing in data/ or outputs/ is modified.
"""

import contextlib
import io
import itertools
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline.config import RunContext, apply_variant, load_config  # noqa: E402
from pipeline.io import read_json, read_jsonl  # noqa: E402
from pipeline.s01_prepare import buggy_code, canonical_code  # noqa: E402
from pipeline.s03_run_models import current_responses  # noqa: E402
from pipeline.s05_execute import build_program, execute_program  # noqa: E402

TID = "Python/132"
CFG = load_config()
ROWS = [json.loads(line) for line in (ROOT / CFG.dataset_path).read_text("utf-8").splitlines()]
ROW = next(r for r in ROWS if r["task_id"] == TID)
TMP = Path(tempfile.mkdtemp())

TEST_CASES = [(eval(m[1]), eval(m[2])) for m in re.finditer(r"assert is_nested\((.+?)\) == (True|False)", ROW["test"])]
DOC_CASES = [(eval(m[1]), eval(m[2])) for m in re.finditer(r"is_nested\((.+?)\) ➞ (True|False)", ROW["instruction"])]


def load_fn(code: str):
    ns: dict = {}
    with contextlib.redirect_stdout(io.StringIO()):  # some answers print their own examples
        exec(code, ns)
    return ns["is_nested"]


def mismatches(code: str, cases: list[tuple[str, bool]]) -> list[str]:
    fn = load_fn(code)
    out = []
    for s, want in cases:
        try:
            got = fn(s)
        except Exception as exc:  # noqa: BLE001
            got = f"EXC {type(exc).__name__}"
        if got != want:
            out.append(f"{s!r}: got {got} want {want}")
    return out


def subsequence_rule(s: str) -> bool:
    """Contains the subsequence '[[]]' (= a valid nested bracket subsequence exists)."""
    j = 0
    for c in s:
        if j < 4 and c == "[[]]"[j]:
            j += 1
    return j == 4


def depth_rule(s: str) -> bool:
    """The typical failing fix: nested = depth reaches 2."""
    d = 0
    for c in s:
        if c == "[":
            d += 1
            if d >= 2:
                return True
        else:
            d = max(0, d - 1)
    return False


def section_task() -> None:
    print("#### 1. TASK")
    for k in ["bug_type", "entry_point", "instruction", "docstring_translated", "declaration",
              "buggy_solution", "canonical_solution", "test"]:
        print(f"===== {k}\n{ROW[k]}")


def section_sanity() -> None:
    print("\n#### 2. SANITY CHECK (same executor/program as step 01)")
    for name, code in [("canonical", canonical_code(ROW)), ("buggy", buggy_code(ROW))]:
        r = execute_program(build_program(code, ROW["test"]), CFG.exec_timeout_s, TMP)
        print(f"{name}: status={r.status} exc={r.exc_type} rc={r.returncode}")
        lines = r.stderr_tail.strip().splitlines()
        print("  stderr:", " | ".join(lines[-3:]) if lines else "(empty)")


def section_spec() -> None:
    print("\n#### 5. CANONICAL vs DOCSTRING EXAMPLES AND TEST ASSERTS")
    canon = canonical_code(ROW)
    print(f"{len(TEST_CASES)} test asserts, {len(DOC_CASES)} docstring examples")
    print("canonical test mismatches:", mismatches(canon, TEST_CASES) or "none")
    print("canonical docstring mismatches:", mismatches(canon, DOC_CASES) or "none")
    fn = load_fn(canon)
    strings = ["".join(t) for n in range(13) for t in itertools.product("[]", repeat=n)]
    print(f"all {len(strings)} bracket strings up to length 12: "
          f"canonical != subsequence-'[[]]' rule on {sum(fn(s) != subsequence_rule(s) for s in strings)}, "
          f"canonical != depth>=2 rule on {sum(fn(s) != depth_rule(s) for s in strings)}")


def section_records() -> None:
    print("\n#### 3/4/6. RECORDS (current responses of each pilot)")
    for variant in (None, "noreason"):
        cfg = apply_variant(CFG, variant)
        ctx = RunContext(cfg, pilot=True, variant=variant, root=ROOT)
        prompts = read_jsonl(ctx.prompts_path)
        cur = current_responses(prompts, cfg.enabled_models, read_jsonl(ctx.responses_path))
        ext = {(r["prompt_id"], r["model"], int(r["run"])): r for r in read_jsonl(ctx.extracted_path)}
        exe = {(r["prompt_id"], r["model"], int(r["run"])): r for r in read_jsonl(ctx.executions_path)}
        task = next(t for t in read_json(ctx.tasks_path)["tasks"] if t["task_id"] == TID)
        label = "noreason" if variant else "1 (effort low)"
        print(f"\n================ PILOT {label}  ({ctx.responses_path.name}, preamble={task['preamble']!r})")
        for key in sorted(k for k in cur if k[0].startswith(TID + "::")):
            resp, e, x = cur[key], ext[key], exe[key]
            code = e["code"]
            rerun = execute_program(build_program(code, ROW["test"], task["preamble"]), CFG.exec_timeout_s, TMP)
            print(f"\n--- {key[0]} {key[1]} run{key[2]}: api={resp['status']} prompt_hash={resp['prompt_hash']} "
                  f"extraction={e['extraction_status']} python_blocks={resp['text'].count('```python')} "
                  f"exec={x['exec_status']} (rerun={rerun.status}) "
                  f"exact_canon={e['exact_match_canonical']} unchanged={e['unchanged_from_buggy']}")
            print(code.rstrip())
            tail = x["stderr_tail"].strip().splitlines()
            print("stderr last lines:", " | ".join(tail[-3:]) if tail else "(none)")
            print("FAILING TEST ASSERTS:", mismatches(code, TEST_CASES) or "none")
            print("FAILING DOCSTRING EXAMPLES:", mismatches(code, DOC_CASES) or "none")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    section_task()
    section_sanity()
    section_spec()
    section_records()
