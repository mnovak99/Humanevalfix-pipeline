# HumanEvalFix: prompt language (EN vs. CS) vs. LLM bug fixing

Research pipeline for the course *Experimental Methods in Software Engineering*.
Research question: does the natural language of the prompt (English vs. Czech)
affect how well an LLM fixes a bug in Python code?

**Experimental invariant:** between language conditions only the instruction and
the docstring *description* change. Code, identifiers and docstring examples
(`>>>` lines and expected outputs) are byte-identical across languages; step 02
asserts this.

## Setup

Requires [uv](https://docs.astral.sh/uv/) (Python 3.12 is installed by uv).

```powershell
uv sync
Copy-Item .env.example .env   # then fill in API keys
uv run pytest
```

## Usage

```powershell
uv run python -m pipeline run --step 01 --pilot
uv run python -m pipeline run --step 03 --dry-run
uv run python -m pipeline run --all --pilot --dry-run
uv run python -m pipeline mock-translations
```

`--pilot` uses `pilot_n_tasks` (a prefix of the same seeded order, so the pilot
set is a subset of the full set) and writes `*_pilot` files. `--dry-run` uses a
fake provider (no API calls) and writes `*_dry` files. Live runs print the call
count and estimated cost and ask for confirmation unless `--yes` is given.

All tunables live in `config.yaml`; API keys in `.env`.

## Pipeline

| Step | Reads | Writes |
|------|-------|--------|
| 01 prepare | HF `bigcode/humanevalpack` (python/test) | `data/tasks.json`, `data/translations_template.xlsx` |
| 02 build prompts | `data/tasks.json`, `data/translations.xlsx` | `data/prompts.jsonl` |
| 03 run models | `data/prompts.jsonl` | `outputs/responses.jsonl` |
| 04 extract | `outputs/responses.jsonl` | `outputs/extracted.jsonl` |
| 05 execute | `outputs/extracted.jsonl` | `outputs/executions.jsonl` |
| 06 aggregate | all of the above | `outputs/results.csv` |

## Dataset

`bigcode/humanevalpack`, config `python`, split `test` (164 tasks), pinned to HF
commit `9a41762f` in `config.yaml`.

Fields used:

| Field | Use |
|-------|-----|
| `task_id` | task key (`Python/N`) |
| `prompt` | imports + (helper functions) + entry-point signature and docstring |
| `buggy_solution` | buggy body; code shown to the model = `prompt + buggy_solution` |
| `canonical_solution` | sanity check only (`prompt + canonical_solution` must pass) |
| `test` | test harness; defines `check(f)` and already ends with `check(<entry_point>)` |
| `entry_point` | function to fix / extract |
| `bug_type` | sampling strata |

Not used: `instruction` (a synthesis instruction, "Write a Python function …",
that embeds the whole English docstring), `docstring`, `declaration`,
`example_test`, `signature`, `failure_symptoms`, `import` and `test_setup`
(the last two are empty for all Python tasks).

Program executed for a test = `preamble + code + test`, where `preamble` is the
prompt without the entry-point function (imports and helper functions).

### Step 01 results (seed 42)

* Sanity check: 164/164 tasks valid (canonical passes, buggy does not), 0 excluded.
  Buggy statuses: 155 `failed`, 6 `error`, 3 `timeout` (Python/10, 156, 160).
* Stratified sample of 100 (proportional, largest remainder): excess logic 19,
  function misuse 5, missing logic 20, operator misuse 15, value misuse 27,
  variable misuse 14. The pilot is the first 20 of the same order and keeps the
  ratios (4/1/4/3/5/3).

### Docstring split

The docstring is split into a translatable **description** and verbatim
**examples**. Example lines are `>>>` sessions (with their expected output) and
call-expression lines such as `f(x) ➞ y`, `f(x) == y`, `* f(x) => y`,
`assert f(x) => y`, including multi-line calls. Each run of example lines is
replaced in the description by a `{EXAMPLE_k}` marker line and re-inserted on
rebuild, so examples are byte-identical across languages by construction.
Indentation and the opening/closing quote layout are structural, not part of the
description. Rebuilding with the English description reproduces the original
code exactly (asserted for every task, including after an `.xlsx` round-trip).

Data written in prose (`Input: [4,2,3]` / `Output: [2, 1]`,
`For num = "AB" the output should be 1.`) and comments between examples are
part of the description: translators translate the words and copy the values.

Helper functions in the prompt (Python/10, 32, 50 in the sample) keep their
English docstrings in every language; they are treated as code.

## Translations and prompts (step 02)

`data/translations.xlsx` follows the contract in the template
(`data/translations_template.xlsx`, sheets `docstrings`, `instructions`,
`guide`). Until real translations exist, `python -m pipeline mock-translations`
writes a mock file (`[CS] ` + EN, status `reviewed`); it refuses to overwrite a
file with real translations unless `--force` is given. Prompts built from mock
text are flagged `mock_translation: true` in `prompts.jsonl`, and live model runs
refuse them.

Validation (errors stop the step, listing every offending task id):
missing/duplicate/unknown task ids, empty cells, status other than `reviewed`,
`docstring_en` not matching `tasks.json`, missing/reordered `{EXAMPLE_k}`
markers, triple quotes, missing or empty instructions, and instruction
placeholders differing from EN (`{entry_point}` is the only one allowed).
Warnings: CS/EN prose length ratio outside 0.6–1.8, values from EN prose
(numbers, quoted strings, `[...]` lists) not found verbatim in the translation,
and new backslashes. A pilot run only needs the pilot rows to be complete.

Prompt = instruction (with `{entry_point}` filled in) + blank line + a fenced
` ```python ` block with the buggy code. **Invariant check:** for every task,
the code with the docstring content removed must be byte-identical to the
original in every language, every example block must appear verbatim and in
order, and the rebuilt English code must equal the original exactly. Any
violation stops the step.

`data/prompts.jsonl` fields: `prompt_id` (`<task_id>::<lang>`), `task_id`,
`language`, `prompt`, `entry_point`, `mock_translation`.

## Model calls (step 03)

Every prompt × enabled model × run (1..`runs_per_prompt`) is one call; each
result is appended to `outputs/responses*.jsonl` immediately. **Resume:** calls
with a successful record are skipped; failed calls are retried on the next run
(the latest record per `(prompt_id, model, run)` wins downstream). Retries use
exponential backoff with jitter for transient errors (429/5xx/network), up to
`retry_max_attempts`; other errors become a `failed` record and the run goes on.
No `temperature`/`top_p`/`top_k` is ever sent; `params` from `config.yaml` are
passed through.

A live (non-dry) run refuses prompts built from mock translations, then prints
the number of calls and the estimated cost (`cost_estimate_tokens` per call ×
`price_per_mtok`) and asks for confirmation unless `--yes`.

`--dry-run` uses a deterministic fake provider (hash of model + prompt) that
returns, in fixed proportions: the canonical solution in a ` ```python ` block,
the buggy code unchanged, the canonical code without a fence, a syntax error, a
wrong function name, an empty response, or a simulated API error — so every
downstream status is exercised.

Response record: `prompt_id, task_id, language, model, provider, run,
timestamp` (UTC ISO), `attempts, status` (`ok`/`failed`), `error, text,
model_reported, input_tokens, output_tokens, reasoning_tokens, latency_s,
raw_meta`.

## Extraction (step 04)

Code = last ` ```python ` (or `py`) block, else the last fenced block of any
kind, else the whole text; an unclosed final fence (truncated output) counts as
a block. The code must `ast.parse` and define a top-level function named
`entry_point`. Statuses: `ok`, `no_code_block` (whole text used and it parsed),
`syntax_error`, `function_missing`, `empty_response`, and `api_error` (the call
itself failed). Usable for execution: `ok`, `no_code_block`.

## Execution (step 05)

Program = `preamble` (the prompt's imports and helper functions) + extracted
code (so the model's own definitions win) + dataset `test`. Each runs in a
fresh `python -I -X utf8` subprocess with `exec_timeout_s`; on timeout the
whole process tree is killed (`taskkill /T` on Windows). Statuses: `passed`,
`failed` (AssertionError), `error` (other exception; type stored), `timeout`,
`skipped` (no usable code). The last 500 characters of stderr are kept.

## Results (step 06)

`outputs/results*.csv`, one row per task × language × model × run:
`task_id, bug_type, language, model, run, extraction_status, exec_status,
passed, input_tokens, output_tokens, reasoning_tokens, latency_s`
(`passed` is 1/0). The step prints pass rate per model × language and status
counts; no statistics.

Dataset quirk: in Python/115 the string literal comes after `import math`, so
it is not a real docstring; it is still treated as the task description.

## Status

- [x] Phase 0: setup, config, CLI skeleton
- [x] Phase 1: s01 + s05 executor (sampling, docstring split, sanity check);
      s05 step wiring follows in Phase 3 together with s04
- [x] Phase 2: mock translations + s02 (current `translations.xlsx` is the MOCK)
- [x] Phase 3: s03 (fake provider) + s04 + s05 wiring + s06; `--all --pilot --dry-run` works end-to-end
- [ ] Phase 4: real providers
