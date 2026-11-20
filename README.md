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

## Dataset fields used

_To be documented in Phase 1 after inspecting the dataset._

## Status

- [x] Phase 0: setup, config, CLI skeleton
- [ ] Phase 1: s01 + s05 (sampling, docstring split, executor, sanity check)
- [ ] Phase 2: mock translations + s02
- [ ] Phase 3: s03 (fake provider) + s04 + s06
- [ ] Phase 4: real providers
