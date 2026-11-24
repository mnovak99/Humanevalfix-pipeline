# Project: Prompt Language vs. LLM Bug Fixing (HumanEvalFix, EN vs. CS)

## Context

University research project (course "Experimental Methods in Software Engineering", team of 3).
Research question: does the natural language of the prompt (English vs. Czech) affect how well an LLM fixes a bug in Python code?

The experiment is a controlled one: **the only thing that changes between conditions is the prompt text (framing sentences + task description).** The code shown to the model (`declaration + buggy_solution`, trailing newlines removed) must be byte-identical in every language, and every `>>>` example with its expected output must be identical in the English and Czech descriptions. Protect this invariant everywhere.

My responsibility is the whole pipeline: dataset → prompts → model calls → code extraction → test execution → `results.csv`. Translations and statistics are done by teammates and are out of scope, except for the dataset contract below.

## Environment

* Windows 11, PowerShell. No Docker. Windows Smart App Control is on: it blocks unsigned, not-yet-reputable native code (this is why `pydantic` is pinned to 2.12.5). Check that new compiled dependencies import before relying on them; never change Windows security settings.
* Python 3.12 from the python.org installer (code-signed, `%LOCALAPPDATA%\Programs\Python\Python312`), used through `uv` (`uv add`, `uv run`). `pyproject.toml` sets `[tool.uv] python-preference = "only-system"`: never let uv download its own CPython, whose unsigned `.pyd` files Smart App Control blocks. Commit `uv.lock`.
* Git: commit after every change set. The GitHub remote `origin` exists; never push without my explicit OK. Never write 'co-authored' in any commit or PR.
* Paths: always use `pathlib`. Read/write all text files as UTF-8 explicitly (Windows defaults to cp1250), with LF line endings (`pipeline/io.py`).

## Repository structure

```
pipeline/
  __main__.py        # CLI entry: python -m pipeline ...
  config.py          # loads config.yaml + .env, per-mode file paths
  dataset.py         # loads data/humanevalpack_cs.jsonl (the only data source)
  validation.py      # translation checks (EN instruction vs. each translation)
  io.py              # UTF-8/LF file helpers
  s01_prepare.py ... s06_aggregate.py
  smoke.py           # EN-only check of the real APIs
  providers/         # anthropic, openai, gemini (disabled), fake (dry-run)
  ui/                # Streamlit web UI (app.py, data.py, runner.py)
tests/
data/
  humanevalpack_cs.jsonl          # dataset (input, from teammates)
  raw/humanevalpack_original.jsonl  # teammates' first version, provenance only, never read
  tasks*.json, prompts*.jsonl     # generated
outputs/             # responses, extracted, executions, results (*.jsonl / *.csv)
config.yaml
start_ui.cmd
```

`.gitignore` includes `.env`, `.venv/`, `__pycache__/`, `outputs/tmp/`, `.claude/`. Data and outputs (except temp files) ARE committed – they are part of the research record.

## Configuration (config.yaml)

Everything tunable lives here, nothing hardcoded in modules: `seed` (42), `n_tasks` (100), `pilot_n_tasks` (20), `languages` ([en, cs]), `runs_per_prompt` (3), `exec_timeout_s`, worker counts, `retry_max_attempts`, `cost_estimate_tokens` (600 in / 2000 out), `dataset_path`, `text_fields`, `prompt_templates`, `models`.

* `text_fields`: language → dataset field with its task description (`en: instruction`, `cs: docstring_translated`).
* `prompt_templates`: one template per language as a YAML literal block (`|-`, never folded `>`), placeholders are dataset fields. Layout (one blank line between sections; description and code each start on their own line):
  * EN: `I gave an AI this prompt, but the result doesn't really work. Could you please fix it?` ⏎⏎ `Used prompt:` ⏎ `{instruction}` ⏎⏎ `Code:` ⏎ `{declaration}` ⏎ `{buggy_solution}` ⏎⏎ `Return the complete corrected function in a single Python code block.`
  * CS: `Dal jsem AI toto zadání, ale výsledek nějak nefunguje. Můžeš to prosím opravit?` ⏎⏎ `Použité zadání:` ⏎ `{docstring_translated}` ⏎⏎ `Kód:` ⏎ `{declaration}` ⏎ `{buggy_solution}` ⏎⏎ `Vrať celou opravenou funkci v jednom bloku kódu v Pythonu.`
  * Every field is inserted with trailing newlines removed (`rstrip("\n")`), nothing else changed, so the template alone sets the spacing (exactly one blank line between sections). The template supplies the line break between `declaration` and `buggy_solution`; step 02 checks that the code shown equals `declaration + buggy_solution` minus trailing newlines. Tests assert the exact Python/23 prompts and that no prompt has `"\n\n\n"` outside the code (25 declarations have PEP 8 blank lines inside, which are task code and stay).
* Models: `claude-haiku-5-5` (anthropic) and `gpt-6-luna` (openai) enabled, `gemini-3.5-flash-lite` disabled. Effort is `low` for every model (`params` passthrough). Never send `temperature`, `top_p` or `top_k`.
* `variants`: named `params` overrides for follow-up runs (`--variant NAME`, UI "Model settings"); outputs get the suffix `_NAME`, tasks and prompts are shared. `noreason`: Haiku `thinking: {type: disabled}` + effort `low` (Haiku 5.5 allows disabled thinking only at effort low/medium/high), GPT `reasoning: {effort: none}`. Never overwrite earlier run outputs: a new configuration gets a new variant.

API keys come from `.env` (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`) via `python-dotenv`. Never print, log or commit keys. `.env.example` has empty values.

## Data

The **only** source is `data/humanevalpack_cs.jsonl`: the 164 HumanEvalPack Python tasks (HumanEvalFix) with all original fields plus `docstring_translated` = the team's reviewed Czech translation of `instruction`. Code/test fields are byte-identical to `bigcode/humanevalpack` (python/test) at HF commit `9a41762f`. Nothing in the pipeline downloads data or reads `data/raw/`.

Fields used: `task_id`, `instruction` (EN text), `docstring_translated` (CS text), `declaration`, `buggy_solution`, `canonical_solution`, `test` (ends with `check(<entry_point>)`), `entry_point`, `bug_type`.

Known limitations (documented in README): `declaration` of Python/10, 32, 38, 50 contains helper functions with English docstrings and Python/64 an English module-level string – identical in both languages, expected. The EN `instruction` of Python/115 contains a Czech word (`Omezení:`); not in the sample.

## Pipeline steps

Each step reads the previous step's file and writes its own. Every step is idempotent and can be run alone.

### s01_prepare

* Load the dataset; sanity check every task on exactly the code the model sees: `declaration + canonical_solution` must pass, `declaration + buggy_solution` must fail. Exclude violators and log them (count + ids) – this goes into the paper's Methods. Currently 164/164 valid.
* Sample `n_tasks` with `seed`, stratified by `bug_type` (proportional, largest remainder), ordered so every prefix stays proportional. `--pilot` = first `pilot_n_tasks` of the same order (pilot ⊂ full set). The task ids must stay the same as the existing sample.
* Save `data/tasks.json`: task records + `buggy_code`, `canonical_code`, `preamble` (imports/helpers), and meta (dataset path + SHA-256, excluded tasks, strata).

### s02_build_prompts

* **Translation checks** for each non-English language against `instruction`; fail hard with the offending task ids: every `>>>` line and the line right after it (expected output) identical and in order; `entry_point` appears in the translation; every backticked identifier from EN appears. Warn only: length ratio outside 0.6–1.8, numbers from EN missing.
* Render one prompt per task and language from `prompt_templates`.
* **Invariant check:** every template contains `{declaration}` + line break + `{buggy_solution}` exactly once and only uses dataset fields; each rendered prompt contains the code (`declaration + buggy_solution` minus trailing newlines) exactly once (so the code part is byte-identical across languages). Fail hard otherwise.
* Output `data/prompts.jsonl`: `prompt_id` (`<task_id>::<lang>`), `task_id`, `language`, `prompt`, `entry_point`.

### s03_run_models

* One interface `ask(model_cfg, prompt) -> Response(text, model_reported, input_tokens, output_tokens, reasoning_tokens, latency_s, raw_meta)`; one implementation per provider using official SDKs; SDK built-in retries disabled. Check current SDK docs for parameter shapes; do not guess.
* Iterate prompt × enabled model × run; append each result to `outputs/responses.jsonl` immediately with UTC timestamp, model settings, `settings_hash` and `prompt_hash`.
* **Resume:** skip a call only if it succeeded with the current prompt text and model settings. Steps 04–06 use only records matching the current prompts and settings, so configurations never mix.
* Retries with `tenacity` (exponential backoff on 429/5xx/network, max 6 attempts); non-retryable errors become failed records.
* **Fake provider** for `--dry-run` (deterministic mix of correct, buggy and broken answers).
* **Live safety:** a real run prints the number of calls and estimated cost and asks for confirmation unless `--yes`.

### s04_extract

* Code = last ```python block, else any fenced block, else whole text; `ast.parse` and find the top-level function `entry_point`.
* Status: `ok`, `no_code_block`, `syntax_error`, `function_missing`, `empty_response`, `api_error`.
* For usable code compare the extracted function after `ast.parse` → `ast.unparse` normalization with docstrings removed: `exact_match_canonical` (vs. `declaration + canonical_solution`), `unchanged_from_buggy` (vs. `declaration + buggy_solution`).

### s05_execute

* Program = preamble + extracted code + dataset `test`; separate subprocess with `exec_timeout_s`; on Windows the whole process tree is killed on timeout. Same executor as s01's sanity check.
* Status: `passed`, `failed` (AssertionError), `error` (exception type stored), `timeout`, `skipped`. Stderr tail (max 500 chars).

### s06_aggregate

* `outputs/results.csv` (UTF-8, comma), one row per task × language × model × run:
`task_id, bug_type, language, model, run, extraction_status, exec_status, passed, exact_match_canonical, unchanged_from_buggy, input_tokens, output_tokens, reasoning_tokens, latency_s`
(`passed` is 1/0 = pass@1 per run, the primary measure; the two comparison columns are 1/0, empty without usable code.)
* Print a short sanity summary. No statistical tests – teammates do that.

## CLI and UI

```
uv run python -m pipeline run --step 01 [--pilot]
uv run python -m pipeline run --step 03 --dry-run
uv run python -m pipeline run --all --pilot --dry-run
uv run python -m pipeline smoke [--n-tasks N] [--dry-run]
uv run python -m pipeline ui          # or double-click start_ui.cmd
```

`--pilot`, `--dry-run`, `--variant NAME` and smoke runs write to separate files (suffixes `_pilot`, `_smoke`, `_NAME`, `_dry`) so test runs never mix with real data. The web UI (127.0.0.1 only) runs the same CLI in a background process and requires an explicit confirmation for paid runs.

## Code style

* English everywhere (code, comments, README, commit messages).
* Comments short and only where logic is non-obvious. No restating what the code says.
* Type hints, small functions, dataclasses for records. Standard `logging`, no print debugging left in.
* `pytest` tests for: translation checks, prompt templates and the invariant check, code extraction and comparison, executor statuses, resume rules, providers (stubbed SDK clients), UI basics.

## Working mode

After each change set: run the tests, commit, show me a short summary and **wait for my review** before continuing. Do not make any paid API call without my explicit OK.

Never modify task code, test code, docstring examples or the dataset file. If something in the dataset looks wrong, report it instead of "fixing" it.
