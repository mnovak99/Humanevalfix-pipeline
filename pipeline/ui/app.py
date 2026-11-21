"""Streamlit UI: run pipeline steps, check the dataset, inspect results.

Start with `uv run python -m pipeline ui` (or double-click start_ui.cmd).
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import streamlit as st

from pipeline.config import ROOT, RunContext, load_config, load_env
from pipeline.ui.data import (
    call_progress,
    cost_estimate,
    dataset_status,
    load_results,
    missing_api_keys,
    pass_rate_table,
    record_details,
    smoke_estimate,
    step_files,
)
from pipeline.ui.runner import Runner, pipeline_args, read_log_tail
from pipeline.validation import ERROR_CHECKS, WARNING_CHECKS

STEPS = {
    "01": "Prepare tasks (dataset, sanity check, sample)",
    "02": "Build prompts (validate translations)",
    "03": "Run models",
    "04": "Extract code",
    "05": "Execute tests",
    "06": "Aggregate results",
}

st.set_page_config(page_title="HumanEvalFix pipeline", layout="wide")


@st.cache_resource
def get_runner() -> Runner:
    # One runner per server process, shared by all browser tabs.
    return Runner()


def load_cfg():
    load_env()
    return load_config()  # re-read on every rerun so config.yaml edits show up


runner = get_runner()
cfg = load_cfg()

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("HumanEvalFix: EN vs CS")
    mode = st.radio(
        "Task set",
        ["Pilot", "Full"],
        format_func=lambda m: f"Pilot ({cfg.pilot_n_tasks} tasks)" if m == "Pilot" else f"Full ({cfg.n_tasks} tasks)",
    )
    dry_run = st.toggle("Dry run (fake model, no cost)", value=True)
    pilot = mode == "Pilot"
    ctx = RunContext(cfg, pilot=pilot, dry_run=dry_run)

    st.caption(f"Files: `*{ctx.results_path.stem.removeprefix('results') or ' (main)'}`")
    st.divider()
    st.subheader("Models")
    for m in cfg.models:
        status = "on" if m.enabled else "off"
        st.markdown(f"**{m.name}** ({m.provider}, {status})  \n`{m.params or '{}'}`")
    st.caption(f"{cfg.runs_per_prompt} runs per prompt, languages: {', '.join(cfg.languages)}")

    @st.fragment(run_every=2)
    def run_status() -> None:
        st.divider()
        run = runner.current
        if run is None:
            st.caption("No run started in this session.")
            return
        icon = {"running": "🔄", "finished": "✅", "stopped": "⏹️"}.get(run.state, "❌")
        st.markdown(f"{icon} **{run.state}**  \n`{run.label}`  \nstarted {run.started:%H:%M:%S}")
        if run.running and st.button("Stop run", type="secondary"):
            runner.stop()
            st.rerun()

    run_status()

overview, run_tab, dataset_tab, results_tab = st.tabs(["Overview", "Run", "Dataset", "Results"])

# --------------------------------------------------------------------------- overview
with overview:
    st.subheader(f"{mode} set, {'dry run' if dry_run else 'live'}")
    files = step_files(ctx)
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "step": f.label,
                    "file": f.path.relative_to(ROOT).as_posix(),
                    "rows": f.rows,
                    "updated": f.modified.strftime("%Y-%m-%d %H:%M") if f.modified else "missing",
                }
                for f in files
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    prog = call_progress(ctx)
    st.markdown(f"**Model calls:** {prog.ok} / {prog.planned} done" + (f", {prog.failed} failed (retried next run)" if prog.failed else ""))
    st.progress(min(1.0, prog.ok / prog.planned) if prog.planned else 0.0)
    ds = dataset_status(cfg, pilot)
    if ds.ok:
        st.success(f"Translation checks pass for {ds.scope}.")
    else:
        st.error("The dataset or its translations have problems (see the Dataset tab).")

# --------------------------------------------------------------------------- run
with run_tab:
    busy = runner.busy()
    if busy:
        st.info("A run is in progress. Its log is below; you can stop it from the sidebar.")

    action = st.radio(
        "What to run",
        ["all", "step", "smoke"],
        format_func={"all": "All steps (01–06)", "step": "One step", "smoke": "Smoke test (English only, a few tasks)"}.get,
        horizontal=True,
    )
    step = None
    n_smoke = 2
    if action == "step":
        step = st.selectbox("Step", list(STEPS), format_func=lambda s: f"{s} {STEPS[s]}")
    if action == "smoke":
        n_smoke = int(st.number_input("Tasks", min_value=1, max_value=cfg.n_tasks, value=2))
        st.caption("Uses the first tasks of the full sample, English prompts only, one run per model. Ignores the task-set choice.")

    calls_models = action in {"all", "smoke"} or step == "03"
    paid = calls_models and not dry_run
    problems: list[str] = []
    confirmed = True
    if paid:
        est = smoke_estimate(cfg, n_smoke) if action == "smoke" else cost_estimate(ctx)
        st.markdown(f"#### Paid run: {est.calls} API calls, estimated **${est.total:.4f}**")
        if est.calls:
            st.dataframe(
                pd.DataFrame([{"model": k, "calls": n, "estimated $": round(c, 4)} for k, (n, c) in est.per_model.items()]),
                hide_index=True,
            )
        tok = cfg.cost_estimate_tokens
        st.caption(f"Estimate assumes {tok.input:.0f} input / {tok.output:.0f} output tokens per call; real output is usually shorter.")
        missing = missing_api_keys(cfg)
        if missing:
            problems.append(f"Missing API keys in .env: {', '.join(missing)}")
        if not dataset_status(cfg, pilot).ok:
            problems.append("Translation checks fail for this task set (see the Dataset tab).")
        if est.calls == 0:
            st.info("All calls for this set are already done; the run only re-runs the other steps.")
        confirmed = st.checkbox(f"I confirm a paid run of {est.calls} calls (~${est.total:.4f})", disabled=bool(problems))
    elif calls_models:
        st.caption("Dry run: the fake provider answers, no API calls, no cost.")

    for p in problems:
        st.error(p)

    if st.button("Start", type="primary", disabled=busy or bool(problems) or not confirmed):
        args = pipeline_args(action, pilot=pilot, dry_run=dry_run, yes=paid and confirmed, step=step, n_tasks=n_smoke)
        runner.start(args)
        st.rerun()

    @st.fragment(run_every=2)
    def live_log() -> None:
        run = runner.current
        if run is None:
            return
        # Fragments refresh alone; when a watched run ends, rerun the page once so
        # the Start button, banners and file overview update too.
        key = str(run.log_path)
        if run.running:
            st.session_state["watching_run"] = key
        elif st.session_state.get("watching_run") == key:
            del st.session_state["watching_run"]
            st.rerun(scope="app")
        st.markdown(f"**Log** · `{run.label}` · {run.state}")
        if "run" in run.args and ("--all" in run.args or "03" in run.args):
            p = call_progress(ctx)
            st.progress(min(1.0, p.ok / p.planned) if p.planned else 0.0, text=f"model calls {p.ok}/{p.planned}")
        st.code(read_log_tail(run.log_path) or "(no output yet)", language=None)

    live_log()

# --------------------------------------------------------------------------- dataset
with dataset_tab:
    ds = dataset_status(cfg, pilot)
    st.markdown(f"**File:** `{ds.path.relative_to(ROOT).as_posix()}`")
    if ds.load_error:
        st.error(f"Cannot load the dataset:\n\n```\n{ds.load_error}\n```")
    else:
        st.caption(f"{ds.n_rows} rows · sha256 `{ds.sha256[:16]}…` · checks cover {ds.scope}")
        fields = ", ".join(f"{lang}: `{f}`" for lang, f in cfg.text_fields.items())
        st.markdown(f"**Text fields:** {fields}")
        report = ds.report
        assert report is not None
        st.dataframe(
            pd.DataFrame(
                [
                    {"check": c, "kind": "error" if c in ERROR_CHECKS else "warning", "tasks": len(report.ids(c)), "task ids": ", ".join(report.ids(c))}
                    for c in (*ERROR_CHECKS, *WARNING_CHECKS)
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        if report.ok:
            st.success("All hard checks pass: step 02 can build the prompts.")
        else:
            st.error("Step 02 will refuse to build prompts until these are fixed in the dataset file.")
        issues = report.errors + report.warnings
        if issues:
            with st.expander(f"Details ({len(issues)} issues)"):
                st.dataframe(
                    pd.DataFrame([{"task": i.task_id, "language": i.language, "check": i.check, "detail": i.detail} for i in issues]),
                    hide_index=True,
                    width="stretch",
                )
        st.caption(
            "Errors: every `>>>` line and the line after it must match the English text; the entry point and all "
            "backticked identifiers must appear. Warnings: length ratio outside 0.6–1.8, English numbers missing."
        )

# --------------------------------------------------------------------------- results
with results_tab:
    source = st.radio("Results of", ["selected task set", "smoke test"], horizontal=True)
    rctx = RunContext(replace(cfg, runs_per_prompt=1), smoke=True, dry_run=dry_run) if source == "smoke test" else ctx
    df = load_results(rctx)
    if df is None:
        st.info(f"No `{rctx.results_path.name}` yet. Run the pipeline first.")
    else:
        st.caption(f"`{rctx.results_path.relative_to(ROOT).as_posix()}` · {len(df)} rows")
        left, right = st.columns(2)
        with left:
            st.markdown("**Pass rate** (model × language)")
            st.dataframe(pass_rate_table(df).style.format("{:.1%}"), width="stretch")
        with right:
            st.markdown("**Status counts**")
            counts = df.groupby(["extraction_status", "exec_status"]).size().rename("rows").reset_index()
            st.dataframe(counts, hide_index=True, width="stretch")

        match_cols = [c for c in ("exact_match_canonical", "unchanged_from_buggy") if c in df]
        if match_cols:
            st.markdown("**Code comparison** (share of rows with usable code; AST-normalized, docstrings ignored)")
            rates = df.groupby(["model", "language"])[match_cols].mean().reset_index()
            st.dataframe(rates.style.format({c: "{:.1%}" for c in match_cols}), hide_index=True, width="stretch")

        f1, f2, f3 = st.columns(3)
        models = f1.multiselect("Model", sorted(df["model"].unique()), default=sorted(df["model"].unique()))
        langs = f2.multiselect("Language", sorted(df["language"].unique()), default=sorted(df["language"].unique()))
        statuses = f3.multiselect("Exec status", sorted(df["exec_status"].unique()), default=sorted(df["exec_status"].unique()))
        view = df[df["model"].isin(models) & df["language"].isin(langs) & df["exec_status"].isin(statuses)]
        st.dataframe(view, hide_index=True, width="stretch", height=300)
        st.download_button("Download results CSV", rctx.results_path.read_bytes(), file_name=rctx.results_path.name)

        st.markdown("#### Inspect one row")
        if view.empty:
            st.caption("No rows match the filters.")
        else:
            labels = [f"{r.task_id} · {r.language} · {r.model} · run {r.run}" for r in view.itertuples()]
            pick = st.selectbox("Row", range(len(view)), format_func=lambda i: labels[i])
            row = view.iloc[pick]
            d = record_details(rctx, row["task_id"], row["language"], row["model"], int(row["run"]))
            resp, ext, exe = d["response"], d["extracted"], d["execution"]
            st.markdown(
                f"**extraction:** {row['extraction_status']} · **execution:** {row['exec_status']}"
                + (f" ({exe.get('exc_type')})" if exe.get("exc_type") else "")
                + f" · tokens in/out/reasoning: {resp.get('input_tokens')}/{resp.get('output_tokens')}/{resp.get('reasoning_tokens')}"
                + f" · same as canonical: {ext.get('exact_match_canonical')} · unchanged from buggy: {ext.get('unchanged_from_buggy')}"
            )
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Prompt**")
                st.code(d["prompt"] or "(missing)", language="markdown")
            with c2:
                st.markdown("**Response**")
                st.code(resp.get("text") or resp.get("error") or "(empty)", language="markdown")
            st.markdown("**Extracted code**")
            st.code(ext.get("code") or "(none)", language="python")
            if exe.get("stderr_tail"):
                st.markdown("**stderr (tail)**")
                st.code(exe["stderr_tail"], language=None)
