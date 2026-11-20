# Project: Prompt Language vs. LLM Bug Fixing (HumanEvalFix, EN vs. CS)

## Context

University research project (course "Experimental Methods in Software Engineering", team of 3).
Research question: does the natural language of the prompt (English vs. Czech) affect how well an LLM fixes a bug in Python code?

The experiment is a controlled one: **the only thing that changes between conditions is the language of the instruction and the docstring description.** The code, identifiers and docstring examples must stay byte-identical across languages. Protect this invariant everywhere.

My responsibility is the whole pipeline: data download → prompts → model calls → code extraction → test execution → `results.csv`. Translations and statistics are done by teammates and are out of scope, except for the file contracts below.

## Environment

* Windows 11, PowerShell. No Docker.
* Python 3.12 managed by `uv` (`uv init`, `uv add`, `uv run`). Commit `uv.lock`.
* Git: initialize a local repo immediately and commit after every phase. A GitHub remote will be added later; do not create or push to any remote. Never write 'co-authored' in any commit or PR.
* Paths: always use `pathlib`. Read/write all text files as UTF-8 explicitly (Windows defaults to cp1250).

## Repository structure

```
pipeline/
  \_\_init\_\_.py
  \_\_main\_\_.py        # CLI entry: python -m pipeline ...
  config.py          # loads config.yaml + .env
  s01\_prepare.py
  s02\_build\_prompts.py
  s03\_run\_models.py
  s04\_extract.py
  s05\_execute.py
  s06\_aggregate.py
  providers/         # anthropic, openai, gemini (optional), fake (dry-run)
  docstrings.py      # split/rebuild docstring (description vs. examples)
tests/
data/                # tasks.json, translations.xlsx, prompts.jsonl
outputs/             # responses.jsonl, extracted.jsonl, executions.jsonl, results.csv
config.yaml
.env.example
README.md
```

`.gitignore` must include `.env`, `.venv/`, `\_\_pycache\_\_/`, `outputs/tmp/`. Data and outputs (except temp files) ARE committed – they are part of the research record.

## Configuration (config.yaml)

Everything tunable lives here, nothing hardcoded in modules:

```yaml
seed: 42
n\_tasks: 100          # full run
pilot\_n\_tasks: 20     # used with --pilot
languages: \[en, cs]
runs\_per\_prompt: 3
exec\_timeout\_s: 10
translations\_path: data/translations.xlsx

models:
  - name: claude-haiku-5-5
    provider: anthropic
    enabled: true
    max\_tokens: 4000
    params: {}        # provider-specific passthrough, e.g. effort setting
    price\_per\_mtok: {input: 0.10, output: 0.50}
  - name: gpt-6-luna
    provider: openai
    enabled: true
    max\_tokens: 4000
    params: {reasoning: {effort: low}}
    price\_per\_mtok: {input: 0.10, output: 0.50}
  - name: gemini-3.5-flash-lite
    provider: gemini
    enabled: false    # fallback only
    max\_tokens: 4000
    params: {}
    price\_per\_mtok: {input: 0.30, output: 2.50}
```

API keys come from `.env` (`ANTHROPIC\_API\_KEY`, `OPENAI\_API\_KEY`, `GEMINI\_API\_KEY`) via `python-dotenv`. Never print, log or commit keys. Provide `.env.example` with empty values.

## Data

Source: Hugging Face dataset `bigcode/humanevalpack`, config `python`, split `test` (164 tasks, HumanEvalFix). Inspect the actual fields before relying on them (expected: `task\_id`, `prompt`, `declaration`, `docstring`, `canonical\_solution`, `buggy\_solution`, `bug\_type`, `entry\_point`, `test`, `example\_test`, `instruction`, ...). Document the fields you use in README.

## Pipeline steps

Each step reads the previous step's file and writes its own. Every step is idempotent and can be run alone.

### s01\_prepare

* Load the dataset, sample `n\_tasks` with `seed`, stratified by `bug\_type` (proportional). Save `data/tasks.json` (full task records + split docstring parts).
* **Sanity check every sampled task:** canonical solution must pass the tests, buggy solution must fail. Exclude tasks that violate this and log them (count + ids) – this goes into the paper's Methods.
* Split each docstring into `description` (prose to translate) and `examples` (`>>>` lines and their expected outputs, kept verbatim). Must round-trip: rebuilding from parts with the original description gives the original code exactly – assert this.
* Export `data/translations\_template.xlsx` (see contract below) with EN filled in and CS empty.
* `--pilot` uses `pilot\_n\_tasks` (first N of the same seeded order, so pilot ⊂ full set).

### Translation file contract (data/translations.xlsx)

Teammates fill this in Google Sheets and export as .xlsx. Sheet `docstrings`:

| task\_id | docstring\_en | docstring\_cs | translated\_by | reviewed\_by | status | note |

Sheet `instructions`: one row per language, columns `language`, `instruction`.

* Status values: `todo`, `translated`, `reviewed`. Only `reviewed` rows are allowed in a live run.
* Adding a language later (e.g. `sk`) = new column `docstring\_sk` + new row in `instructions` + entry in `languages`. No code changes.
* **Until real translations exist:** a command `python -m pipeline mock-translations` creates a mock `translations.xlsx` where CS = `"\[CS] " + EN` and status = `reviewed`, so the whole pipeline can be tested end-to-end. Switching to real translations = replacing the file. Live runs must refuse to run if any CS text starts with `\[CS] `.

### s02\_build\_prompts

* Load and validate translations: all task\_ids present and no extras, no empty cells, all `reviewed` (else fail with a list of offending ids), warn if CS/EN length ratio is outside 0.6–1.8.
* Build the code shown to the model: buggy function with the docstring description replaced by the target-language text, examples unchanged.
* **Invariant check:** the code with the docstring removed must be identical across languages. Fail hard otherwise.
* Prompt = `instruction` + blank line + fenced ```python code block. Same template for every language.
* Output `data/prompts.jsonl`: `prompt\_id`, `task\_id`, `language`, `prompt`, `entry\_point`.

### s03\_run\_models

* One interface `ask(model\_cfg, prompt) -> Response(text, model\_reported, input\_tokens, output\_tokens, reasoning\_tokens, latency\_s, raw\_meta)`; one implementation per provider using official SDKs (`anthropic`, `openai`, `google-genai`). Pass `params` from config through. **Never send `temperature`, `top\_p` or `top\_k`** – current models reject or ignore them. Check current SDK docs for the exact effort/reasoning parameter shape; do not guess.
* Iterate prompt × enabled model × run (1..runs\_per\_prompt). Append each result to `outputs/responses.jsonl` immediately, with timestamp (UTC ISO) and the configured model id.
* **Resume:** on start, load existing (prompt\_id, model, run) keys and skip them.
* Retries with `tenacity`: exponential backoff on 429/5xx/network errors, max 6 attempts; non-retryable errors are logged as a failed record, not crash the run.
* **Fake provider** for `--dry-run`: returns the canonical solution in a code block for some prompts and garbage for others (deterministic), so steps 04–06 can be tested without API calls.
* **Live safety:** a real (non-dry) run first prints the number of calls and estimated cost from `price\_per\_mtok` (assume 600 input / 2000 output tokens per call) and asks for confirmation unless `--yes`.
* Show progress with `tqdm`.

### s04\_extract

* Extract code from the response: last ```python block, else any fenced block, else whole text.
* `ast.parse` it and find the function named `entry\_point`.
* Status: `ok`, `no\_code\_block` (used whole text and it parsed), `syntax\_error`, `function\_missing`, `empty\_response`. Keep extracted code in `outputs/extracted.jsonl`.

### s05\_execute

* For each extracted record with usable code: write a temp file (`outputs/tmp/`) = needed imports + extracted code + the dataset `test` + the call that runs it (verify how `check(...)` is invoked in this dataset).
* Run with `subprocess.run(\[sys.executable, file], timeout=exec\_timeout\_s, capture\_output=True)`. On Windows make sure a timed-out process (and its children) is actually killed.
* Status: `passed`, `failed` (AssertionError), `error` (other exception; store exception type), `timeout`, `skipped` (no usable code). Save stderr tail (max 500 chars) for debugging.
* The same executor is used by s01's sanity check – one implementation.
* Execution may run in parallel (`concurrent.futures`, small pool).

### s06\_aggregate

* Write `outputs/results.csv` (UTF-8, comma), one row per task × language × model × run:
`task\_id, bug\_type, language, model, run, extraction\_status, exec\_status, passed, input\_tokens, output\_tokens, reasoning\_tokens, latency\_s`
(`passed` is 1/0.)
* Print a short sanity summary: pass rate per model × language, count of extraction failures. No statistical tests – teammates do that.

## CLI

```
uv run python -m pipeline run --step 01 \[--pilot]
uv run python -m pipeline run --step 03 --dry-run
uv run python -m pipeline run --all --pilot --dry-run
uv run python -m pipeline mock-translations
```

Use `argparse` or `typer`. `--pilot` and `--dry-run` must write to separate output files (e.g. suffix `\_pilot`, `\_dry`) so test runs never mix with real data.

## Code style

* English everywhere (code, comments, README, commit messages).
* Comments short and only where logic is non-obvious. No restating what the code says.
* Type hints, small functions, dataclasses for records. Standard `logging`, no print debugging left in.
* `pytest` tests for: docstring split/rebuild round-trip, invariant check in s02, code extraction cases, executor statuses (pass/fail/error/timeout).

## Working mode

Work in phases. After each phase: run the tests, commit, show me a short summary and **wait for my review** before continuing.

1. **Phase 0 – setup:** uv project, dependencies, git init, `.gitignore`, `.env.example`, config, CLI skeleton.
2. **Phase 1 – s01 + s05:** dataset loading, sampling, docstring split, executor, sanity check (canonical passes / buggy fails). Report how many tasks passed the check.
3. **Phase 2 – mock translations + s02.**
4. **Phase 3 – s03 (fake provider first) + s04 + s06:** full `--all --pilot --dry-run` run must work end-to-end.
5. **Phase 4 – real providers:** implement Anthropic and OpenAI (Gemini optional, disabled). Do not make any paid API call without my explicit OK.

Never modify task code, test code or docstring examples. If something in the dataset looks wrong, report it instead of "fixing" it.

