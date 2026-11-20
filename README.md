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

Dataset quirk: in Python/115 the string literal comes after `import math`, so
it is not a real docstring; it is still treated as the task description.

## Status

- [x] Phase 0: setup, config, CLI skeleton
- [x] Phase 1: s01 + s05 executor (sampling, docstring split, sanity check);
      s05 step wiring follows in Phase 3 together with s04
- [ ] Phase 2: mock translations + s02
- [ ] Phase 3: s03 (fake provider) + s04 + s06
- [ ] Phase 4: real providers
