# Python/132 (`is_nested`): failure analysis

**Why:** Python/132 failed in all 12 runs of the noreason pilot (both models, both languages). Luna CS also failed once in the first pilot. We checked whether these failures are model errors or a problem in the dataset or the pipeline.

**Verdict: genuine model error.** The task is hard to read but consistent. The pipeline handled every record correctly.

## Evidence

- **Bug and canonical fix:** one character, `'('` → `'['`. The sanity check gives canonical `passed` and buggy `failed` (first assert, `'[[]]'`).
- **Spec, tests and canonical agree:**
  - The canonical solution passes all 14 test asserts and all 6 docstring examples.
  - On all 8,191 bracket strings up to length 12, the canonical solution is exactly the rule "contains the subsequence `[[]]`", i.e. a *valid* nested subsequence exists.
- **What models changed:** no model made the one-character fix (`exact_match_canonical` = 0 in all 24 records). All rewrote the algorithm.
  - **First pilot:** 11/12 passed, all using the subsequence-`[[]]` rule (or an equivalent stack rule).
  - **All 13 failures:** one shared misreading, "nested = depth reaches 2". It ignores the word *valid*: the outer bracket must also close. This rule differs from the canonical on 1,623 of the 8,191 strings.
    - Every failing fix returns True for `'[[]'`, where False is expected.
    - 10 of the 13 also contradict a docstring example the model saw in its prompt: `'[]]]]]]][[[[[]'` → False.
- **Pipeline checks:**
  - **Prompts:** both pilots used identical prompts (EN `a986b754588b`, CS `2264fb6c8d0d`).
  - **Extraction:** each response had one python block, and the full function was extracted.
  - **Harness:** the preamble is only whitespace (no missing imports or helpers). Re-running every program reproduces the stored `exec_status`.
- **Language:** EN and CS fail the same way. The effect comes from switching reasoning off, not from the prompt language.

Caveat for the paper: the HumanEval wording ("at least one bracket in the subsequence is nested") is terse, and the docstring has no unclosed example like `'[[]'`. This makes the task hard, but it does not make it inconsistent with the tests.

## Files

- `analyze132.py`: read-only script (no API calls). Run it from the repo root: `uv run python analysis/python132/analyze132.py > analysis/python132/out132.txt`
- `out132.txt`: its output. It contains the task fields, the sanity check, the spec checks, and every record's code, `exec_status`, stderr tail, and failing inputs.
