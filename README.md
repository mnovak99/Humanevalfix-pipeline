# HumanEvalFix: prompt language (EN vs. CS) vs. LLM bug fixing

Research pipeline for the course *Experimental Methods in Software Engineering*.
Research question: does the natural language of the prompt (English vs. Czech)
affect how well an LLM fixes a bug in Python code?

**Experimental invariant:** between the language conditions only the prompt text
changes (the framing sentences and the task description). The code shown to the
model (`declaration + buggy_solution`) is byte-identical in every language, and
every `>>>` example with its expected output is identical in the English and Czech
descriptions; step 02 checks both and stops otherwise.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and an installed **Python 3.12 from
python.org** (code-signed). uv is configured with `python-preference = "only-system"`
and does not download its own interpreter: uv's managed CPython builds are unsigned,
and Windows Smart App Control blocks their files (seen for `_multiprocessing.pyd` and
`pyexpat.pyd`, which the web UI needs).

```powershell
winget install --id Python.Python.3.12 -e --source winget --scope user
uv sync
Copy-Item .env.example .env   # then fill in API keys
uv run pytest
```

## Web UI

Double-click `start_ui.cmd` (or run `uv run python -m pipeline ui`). The UI opens at
http://127.0.0.1:8501 and is reachable from this computer only.

* **Overview**: which step files exist for the chosen task set (pilot/full, dry/live),
  model-call progress, whether the translation checks pass.
* **Run**: run all steps, one step, or the smoke test. Runs are background processes with a
  live log; they keep going if the browser tab closes, and can be stopped from the sidebar.
  A live run shows the call count and estimated cost and starts only after you tick the
  confirmation box; it is blocked while API keys are missing or translation checks fail.
* **Dataset**: the dataset file, its hash, and the translation-check report (errors,
  warnings, offending task ids, details) for the selected task set.
* **Results**: pass rate and code-comparison rates per model × language, status counts, a
  filterable table, CSV download, and an inspector showing prompt, response, extracted
  code and stderr per row.

Close the console window to stop the UI; stop a running pipeline run first.

## Usage (command line)

```powershell
uv run python -m pipeline run --step 01 --pilot
uv run python -m pipeline run --step 03 --dry-run
uv run python -m pipeline run --all --pilot --dry-run
uv run python -m pipeline smoke              # EN-only check of the real APIs
```

`--pilot` uses `pilot_n_tasks` (a prefix of the same seeded order, so the pilot
set is a subset of the full set) and writes `*_pilot` files. `--dry-run` uses a
fake provider (no API calls) and writes `*_dry` files. Live runs print the call
count and estimated cost and ask for confirmation unless `--yes` is given.

All tunables live in `config.yaml`; API keys in `.env`.

## Pipeline

| Step | Reads | Writes |
|------|-------|--------|
| 01 prepare | `data/humanevalpack_cs.jsonl` | `data/tasks.json` |
| 02 build prompts | `data/tasks.json`, templates in `config.yaml` | `data/prompts.jsonl` |
| 03 run models | `data/prompts.jsonl` | `outputs/responses.jsonl` |
| 04 extract | `outputs/responses.jsonl`, `data/tasks.json` | `outputs/extracted.jsonl` |
| 05 execute | `outputs/extracted.jsonl` | `outputs/executions.jsonl` |
| 06 aggregate | all of the above | `outputs/results.csv` |

## Dataset

`data/humanevalpack_cs.jsonl` is the pipeline's **only** data source: the 164
HumanEvalPack Python tasks (HumanEvalFix) with all original fields plus
`docstring_translated`, the team's reviewed Czech translation of `instruction`.
The code and test fields (`prompt`, `declaration`, `canonical_solution`,
`buggy_solution`, `test`) are byte-identical to `bigcode/humanevalpack`
(config `python`, split `test`) at HF commit `9a41762f`; this was checked once
when the file arrived.

`data/raw/humanevalpack_original.jsonl` is the teammates' first version, kept for
provenance and not read by the pipeline. It differs from the final file only in
`docstring_translated` of 8 tasks (Python/10, 32, 37, 40, 49, 53, 81, 99), where
review fixed translated function names in `>>>` lines, decimal commas, a wrong
expected output and typos. The step-02 checks flag 7 of these as errors and the
eighth (decimal commas in a table, Python/81) as a warning.

Fields used:

| Field | Use |
|-------|-----|
| `task_id` | task key (`Python/N`) |
| `instruction` | English task description (in the EN prompt) |
| `docstring_translated` | Czech task description (in the CS prompt) |
| `declaration` | imports, helper functions and the entry-point signature (ends with a newline) |
| `buggy_solution` | buggy body; code shown to the model = `declaration + buggy_solution` |
| `canonical_solution` | sanity check and `exact_match_canonical` |
| `test` | test harness; defines `check(f)` and already ends with `check(<entry_point>)` |
| `entry_point` | function to fix / extract |
| `bug_type` | sampling strata |

Not used: `prompt`, `docstring`, `text`, `example_test`, `signature`, `failure_symptoms`.

Program executed for a test = `preamble + code + test`, where `preamble` is
`declaration + buggy_solution` without the entry-point function (imports and
helper functions).

### Known limitations of the data

* **English text inside the code part.** In Python/10, 32, 38 and 50 `declaration`
  contains helper functions with English docstrings, and in Python/64 a module-level
  string `FIX = """Add more test cases."""`. This text is part of the code, so it is
  identical (English) in both language conditions. Python/10, 32 and 50 are in the
  100-task sample (Python/32 also in the pilot).
* **Czech word in an English description.** The `instruction` of Python/115 says
  `Omezení:` instead of the original `Constraints:`. Python/115 is not in the sample.
* `docstring` (not used) differs from HF in Python/41 (line break) and Python/142
  (a stray `"` removed).

### Step 01 results (seed 42)

* Sanity check on exactly the code the model sees: `declaration + canonical_solution`
  must pass and `declaration + buggy_solution` must fail. 164/164 tasks valid, 0
  excluded. Buggy statuses: 155 `failed`, 6 `error`, 3 `timeout` (Python/10, 156, 160).
* Stratified sample of 100 (proportional, largest remainder): excess logic 19,
  function misuse 5, missing logic 20, operator misuse 15, value misuse 27,
  variable misuse 14. The pilot is the first 20 of the same order and keeps the
  ratios (4/1/4/3/5/3). The task ids are the same as in the earlier HF-based sample.
* `tasks.json` records the dataset path and its SHA-256.

## Prompts and translation checks (step 02)

One template per language in `config.yaml` (`prompt_templates`); placeholders are
dataset fields:

```text
I gave an AI this prompt, but the result doesn't really work. Could you please fix it?

Used prompt:
{instruction}

Code:
{declaration}
{buggy_solution}

Return the complete corrected function in a single Python code block.
```

```text
Dal jsem AI toto zadání, ale výsledek nějak nefunguje. Můžeš to prosím opravit?

Použité zadání:
{docstring_translated}

Kód:
{declaration}
{buggy_solution}

Vrať celou opravenou funkci v jednom bloku kódu v Pythonu.
```

The templates are YAML literal blocks (`|-`), so line breaks and blank lines are kept
exactly; a test checks the rendered Python/23 prompts character by character.

Every field is inserted with its **trailing newlines removed** (`rstrip("\n")`;
nothing else inside a field changes), so the template alone sets the spacing: exactly
one blank line between sections, and the description and the code each start on their
own line. Because `declaration` loses its final newline, the template puts the line
break between `{declaration}` and `{buggy_solution}` back; step 02 checks for every task
that the code shown equals `declaration + buggy_solution` minus trailing newlines.
Before stripping, trailing newlines produced double blank lines in Python/3 (CS
`docstring_translated`) and in Python/99, 117, 153 (`buggy_solution`, both languages).

A test asserts that no prompt contains `"\n\n\n"` outside the code. Inside the code it
can occur: 25 `declaration` fields (15 in the sample) have PEP 8 blank lines between
the imports and the function, e.g. Python/0 `from typing import List\n\n\ndef ...`;
this is task code and is left unchanged (identical in both languages). Trailing spaces
are kept too (e.g. `buggy_solution` of Python/76 ends with `" \n"` → the code's last
line ends with a space).

`text_fields` maps each language to the field with its description
(`en: instruction`, `cs: docstring_translated`). Adding a language = a new field in
the dataset, an entry in `languages`, `text_fields` and `prompt_templates`.

**Translation checks** (each non-English text against `instruction`; offending
task ids are printed):

| Check | Kind |
|-------|------|
| every `>>>` line and the line right after it (expected output) identical, in order | error |
| `entry_point` appears in the translation | error |
| every backticked identifier from EN (e.g. the signature) appears | error |
| length ratio translation/EN outside 0.6–1.8 | warning |
| numbers from EN missing in the translation | warning |

Current result: no errors; 2 warnings over all 164 tasks (Python/93, 123: numbers
written out in words), 1 in the 100-task sample (Python/123), none in the pilot.

**Invariant check:** every template must contain `{declaration}` + line break +
`{buggy_solution}` exactly once, and each rendered prompt must contain the code
(`declaration + buggy_solution` without trailing newlines) exactly once, so the code
part is byte-identical across languages. Any violation stops the step.

`data/prompts.jsonl` fields: `prompt_id` (`<task_id>::<lang>`), `task_id`,
`language`, `prompt`, `entry_point`.

## Model calls (step 03)

Every prompt × enabled model × run (1..`runs_per_prompt`) is one call; each
result is appended to `outputs/responses*.jsonl` immediately. Retries use
exponential backoff with jitter for transient errors (429/5xx/network), up to
`retry_max_attempts`; other errors become a `failed` record and the run goes on.
No `temperature`/`top_p`/`top_k` is ever sent; `params` from `config.yaml` are
passed through. Effort is `low` for every model.

Each record stores the model `settings` it ran with (provider, `max_tokens`,
`params`), a `settings_hash` and a `prompt_hash`. **Resume** skips a call only if it
succeeded with the current prompt text **and** settings; after a template or config
change the call is made again. Steps 04–06 only use records that match the current
prompts and settings, so results never mix configurations (older records stay in
the file as history).

A live (non-dry) run prints the number of calls and the estimated cost
(`cost_estimate_tokens` per call × `price_per_mtok`) and asks for confirmation unless
`--yes`.

`--dry-run` uses a deterministic fake provider (hash of model + prompt) that
returns, in fixed proportions: the canonical solution in a ` ```python ` block,
the buggy code unchanged, the canonical code without a fence, a syntax error, a
wrong function name, an empty response, or a simulated API error — so every
downstream status is exercised.

Response record: `prompt_id, task_id, language, model, provider, run, settings,
settings_hash, prompt_hash, timestamp` (UTC ISO), `attempts, status`
(`ok`/`failed`), `error, text, model_reported, input_tokens, output_tokens,
reasoning_tokens, latency_s, raw_meta`.

### Providers

| Provider | SDK / endpoint | `params` passthrough | `reasoning_tokens` from |
|----------|----------------|----------------------|-------------------------|
| anthropic | `anthropic` 2.x, `messages.create` | request fields, e.g. `output_config: {effort: low}` | `usage.output_tokens_details.thinking_tokens` |
| openai | `openai` 3.x, `responses.create` (`store=False`) | request fields, e.g. `reasoning: {effort: low}` | `usage.output_tokens_details.reasoning_tokens` |
| gemini | `google-genai`, `models.generate_content` | `GenerateContentConfig` fields | `usage_metadata.thoughts_token_count` |

The prompt is sent as a single user message; no system prompt. SDK built-in
retries are disabled (`max_retries=0`) so s03 owns the retry policy. Retryable:
connection errors/timeouts and HTTP 408/409/429/5xx/529; everything else is a
failed record. `output_tokens` includes thinking/reasoning tokens. Model-side
metadata (stop reason / status, refusal details, request id) goes to `raw_meta`.

### Smoke test (real APIs, English only)

```powershell
uv run python -m pipeline smoke              # 2 tasks x enabled models, 1 run each
uv run python -m pipeline smoke --n-tasks 5
uv run python -m pipeline smoke --dry-run    # same path with the fake provider
```

Checks API keys, model ids and `params` before a paid experiment run. It takes the
first N tasks of `data/tasks.json`, builds the same English prompts as step 02
(tested to be identical) and runs steps 03–06 with one run per prompt. Same cost
confirmation as a live run. Files: `data/prompts_smoke.jsonl`, `outputs/*_smoke.*`;
they never mix with research data.

## Extraction (step 04)

Code = last ` ```python ` (or `py`) block, else the last fenced block of any
kind, else the whole text; an unclosed final fence (truncated output) counts as
a block. The code must `ast.parse` and define a top-level function named
`entry_point`. Statuses: `ok`, `no_code_block` (whole text used and it parsed),
`syntax_error`, `function_missing`, `empty_response`, and `api_error` (the call
itself failed). Usable for execution: `ok`, `no_code_block`.

For usable code the step also compares the extracted entry-point function with the
references after `ast.parse` → `ast.unparse` normalization, with docstrings removed
(so formatting, comments and docstrings do not matter): `exact_match_canonical`
(equals `declaration + canonical_solution`) and `unchanged_from_buggy` (equals
`declaration + buggy_solution`).

## Execution (step 05)

Program = `preamble` + extracted code (so the model's own definitions win) +
dataset `test`. Each runs in a fresh `python -I -X utf8` subprocess with
`exec_timeout_s`; on timeout the whole process tree is killed (`taskkill /T` on
Windows). Statuses: `passed`, `failed` (AssertionError), `error` (other exception;
type stored), `timeout`, `skipped` (no usable code). The last 500 characters of
stderr are kept.

## Results (step 06)

`outputs/results*.csv`, one row per task × language × model × run:

`task_id, bug_type, language, model, run, extraction_status, exec_status, passed,
exact_match_canonical, unchanged_from_buggy, input_tokens, output_tokens,
reasoning_tokens, latency_s`

The primary measure is test-based: `passed` (1/0; pass@1 per run).
`exact_match_canonical` and `unchanged_from_buggy` are 1/0, and empty when there was
no usable code. The step prints pass rate per model × language, status counts and
the two comparison counts; no statistics.

## Status

- [x] Phases 0–4: setup, sampling + executor, prompts, model runner (fake + real
      providers), extraction, execution, aggregation
- [x] Web UI (Streamlit), effort `low` for all models, EN-only smoke test
- [x] Final dataset (`data/humanevalpack_cs.jsonl`) with new prompt templates,
      translation checks and code-comparison columns; no paid calls made yet

Notes on Windows Smart App Control (keep these unless the pipeline runs elsewhere, e.g.
WSL/Linux): the interpreter is the signed python.org Python 3.12 (3.12.10), not a
uv-managed build, and `pydantic` is pinned to 2.12.5 because the newer `pydantic-core`
DLL was blocked.
