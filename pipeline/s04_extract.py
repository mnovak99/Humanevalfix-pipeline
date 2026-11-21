"""Extract the entry-point function from model responses (outputs/extracted.jsonl)."""

from __future__ import annotations

import ast
import logging
import re
from collections import Counter
from dataclasses import dataclass

from pipeline.config import RunContext
from pipeline.io import read_json, read_jsonl, write_jsonl
from pipeline.s03_run_models import OK as RESPONSE_OK
from pipeline.s03_run_models import current_responses, latest_by_key

log = logging.getLogger(__name__)

OK, NO_CODE_BLOCK, SYNTAX_ERROR, FUNCTION_MISSING, EMPTY_RESPONSE, API_ERROR = (
    "ok",
    "no_code_block",
    "syntax_error",
    "function_missing",
    "empty_response",
    "api_error",
)
USABLE = {OK, NO_CODE_BLOCK}

# Opening fence with optional language tag; the closing fence may be missing (truncated output).
_FENCE_RE = re.compile(r"^[ \t]*```[ \t]*([\w+\-.]*)[^\n]*\n(.*?)(?:^[ \t]*```[ \t]*$|\Z)", re.DOTALL | re.MULTILINE)
_PYTHON_TAGS = {"python", "py", "python3"}


@dataclass(frozen=True)
class Extraction:
    status: str
    code: str | None
    n_blocks: int
    source: str  # "python_block", "any_block", "whole_text", "none"


def _strip_docstrings(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]


def normalized_function(code: str, entry_point: str) -> str | None:
    """ast.unparse of the top-level entry-point function without docstrings (None if absent).

    Normalizes formatting, comments and quote style, so equal output means the same code.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    fn = next(
        (n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == entry_point),
        None,
    )
    if fn is None:
        return None
    _strip_docstrings(fn)
    return ast.unparse(fn)


def compare_to_references(code: str | None, task: dict) -> tuple[bool | None, bool | None]:
    """(exact_match_canonical, unchanged_from_buggy); None when there is no usable code."""
    if code is None:
        return None, None
    got = normalized_function(code, task["entry_point"])
    if got is None:
        return None, None
    canonical = normalized_function(task["declaration"] + task["canonical_solution"], task["entry_point"])
    buggy = normalized_function(task["declaration"] + task["buggy_solution"], task["entry_point"])
    return got == canonical, got == buggy


def find_blocks(text: str) -> list[tuple[str, str]]:
    return [(m.group(1).lower(), m.group(2)) for m in _FENCE_RE.finditer(text)]


def _has_function(tree: ast.Module, name: str) -> bool:
    return any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name for n in tree.body)


def extract(text: str | None, entry_point: str) -> Extraction:
    if text is None or not text.strip():
        return Extraction(EMPTY_RESPONSE, None, 0, "none")
    blocks = find_blocks(text)
    python_blocks = [code for tag, code in blocks if tag in _PYTHON_TAGS]
    if python_blocks:
        code, source = python_blocks[-1], "python_block"
    elif blocks:
        code, source = blocks[-1][1], "any_block"
    else:
        code, source = text, "whole_text"

    code = code.strip("\n") + "\n"
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return Extraction(SYNTAX_ERROR, code, len(blocks), source)
    if not _has_function(tree, entry_point):
        return Extraction(FUNCTION_MISSING, code, len(blocks), source)
    return Extraction(NO_CODE_BLOCK if source == "whole_text" else OK, code, len(blocks), source)


def run(ctx: RunContext) -> None:
    if not ctx.responses_path.exists():
        raise FileNotFoundError(f"{ctx.responses_path} not found; run step 03 first")
    prompt_list = read_jsonl(ctx.prompts_path)
    prompts = {p["prompt_id"]: p for p in prompt_list}
    tasks = {t["task_id"]: t for t in read_json(ctx.tasks_path)["tasks"]}
    all_responses = read_jsonl(ctx.responses_path)
    responses = current_responses(prompt_list, ctx.config.enabled_models, all_responses)
    stale = len(latest_by_key(all_responses)) - len(responses)
    if stale:
        log.info("ignoring %d older records (other prompt text, model or settings)", stale)

    out = []
    for (prompt_id, model, run_no), resp in responses.items():
        if resp["status"] != RESPONSE_OK:
            ex = Extraction(API_ERROR, None, 0, "none")
        else:
            ex = extract(resp["text"], prompts[prompt_id]["entry_point"])
        usable = ex.code if ex.status in USABLE else None
        exact, unchanged = compare_to_references(usable, tasks[resp["task_id"]])
        out.append(
            {
                "prompt_id": prompt_id,
                "task_id": resp["task_id"],
                "language": resp["language"],
                "model": model,
                "run": run_no,
                "extraction_status": ex.status,
                "extraction_source": ex.source,
                "n_blocks": ex.n_blocks,
                "exact_match_canonical": exact,
                "unchanged_from_buggy": unchanged,
                "code": ex.code,
            }
        )
    write_jsonl(ctx.extracted_path, out)
    counts = Counter(r["extraction_status"] for r in out)
    log.info("extracted %d responses -> %s: %s", len(out), ctx.extracted_path, dict(sorted(counts.items())))
