"""Extract the entry-point function from model responses (outputs/extracted.jsonl)."""

from __future__ import annotations

import ast
import logging
import re
from collections import Counter
from dataclasses import dataclass

from pipeline.config import RunContext
from pipeline.io import read_jsonl, write_jsonl
from pipeline.s03_run_models import OK as RESPONSE_OK
from pipeline.s03_run_models import latest_by_key

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
    prompts = {p["prompt_id"]: p for p in read_jsonl(ctx.prompts_path)}
    responses = latest_by_key(read_jsonl(ctx.responses_path))

    out = []
    for (prompt_id, model, run_no), resp in responses.items():
        if resp["status"] != RESPONSE_OK:
            ex = Extraction(API_ERROR, None, 0, "none")
        else:
            ex = extract(resp["text"], prompts[prompt_id]["entry_point"])
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
                "code": ex.code,
            }
        )
    write_jsonl(ctx.extracted_path, out)
    counts = Counter(r["extraction_status"] for r in out)
    log.info("extracted %d responses -> %s: %s", len(out), ctx.extracted_path, dict(sorted(counts.items())))
