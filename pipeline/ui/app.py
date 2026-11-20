"""Streamlit UI: run pipeline steps, manage translations, inspect results.

Start with `uv run python -m pipeline ui` (or double-click start_ui.cmd).
"""

from __future__ import annotations

import shutil
from dataclasses import replace
from datetime import datetime

import pandas as pd
import streamlit as st

from pipeline.config import ROOT, RunContext, load_config, load_env
from pipeline.translations import TranslationError, write_mock
from pipeline.ui.data import (
    call_progress,
    cost_estimate,
    load_results,
    missing_api_keys,
    pass_rate_table,
    record_details,
    smoke_estimate,
    step_files,
    translation_status,
)
from pipeline.ui.runner import Runner, pipeline_args, read_log_tail

STEPS = {
    "01": "Prepare tasks (dataset, sanity check, sample)",
    "02": "Build prompts (validate translations)",
    "03": "Run models",
    "04": "Extract code",
    "05": "Execute tests",
    "06": "Aggregate results",
}
BACKUP_DIR = ROOT / "outputs" / "tmp" / "translations_backups"

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

overview, run_tab, trans_tab, results_tab = st.tabs(["Overview", "Run", "Translations", "Results"])

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
        use_container_width=True,
    )
    prog = call_progress(ctx)
    st.markdown(f"**Model calls:** {prog.ok} / {prog.planned} done" + (f", {prog.failed} failed (retried next run)" if prog.failed else ""))
    st.progress(min(1.0, prog.ok / prog.planned) if prog.planned else 0.0)
    tr = translation_status(cfg, pilot)
    if not tr.exists:
        st.warning("No translations file yet (see the Translations tab).")
    elif tr.is_mock:
        st.warning("Translations are the MOCK version: only dry runs are possible.")
    elif tr.errors:
        st.error("Translations are not valid yet (see the Translations tab).")
    else:
        st.success("Translations are valid for this task set.")

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
        if action != "smoke":
            tr = translation_status(cfg, pilot)
            if not tr.usable_live:
                problems.append("Live runs need real, valid, reviewed translations (see the Translations tab).")
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

# --------------------------------------------------------------------------- translations
with trans_tab:
    tr = translation_status(cfg, pilot)
    path = cfg.translations_path
    if not tr.exists:
        st.warning(f"`{path.relative_to(ROOT).as_posix()}` does not exist yet.")
    else:
        kind = "MOCK ([CS] + English)" if tr.is_mock else "real"
        st.markdown(f"**File:** `{path.relative_to(ROOT).as_posix()}` ({kind})")
        st.markdown("**Rows by status:** " + ", ".join(f"{k}: {v}" for k, v in tr.status_counts.items()))
        if tr.errors:
            st.error(f"Not valid for the {mode.lower()} set:\n\n```\n{tr.errors}\n```")
        else:
            st.success(f"Valid for the {mode.lower()} set.")
        if tr.warnings:
            with st.expander(f"{len(tr.warnings)} warnings (check, not blocking)"):
                st.code("\n".join(tr.warnings), language=None)

    st.divider()
    st.markdown("#### Replace with the translators' file")
    st.caption("Export the Google Sheet as .xlsx. The current file is backed up to outputs/tmp/translations_backups/ first.")
    upload = st.file_uploader("translations.xlsx", type=["xlsx"])
    if upload is not None and st.button("Replace translations file", type="primary"):
        try:
            sheets = pd.read_excel(upload, sheet_name=None, dtype=str, keep_default_na=False)
            missing_sheets = {"docstrings", "instructions"} - set(sheets)
            if missing_sheets:
                raise TranslationError(f"the file has no sheet(s) {sorted(missing_sheets)}")
            if path.exists():
                BACKUP_DIR.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, BACKUP_DIR / f"translations-{datetime.now():%Y%m%d-%H%M%S}.xlsx")
            path.write_bytes(upload.getvalue())
            st.success("Replaced. Run step 02 to rebuild the prompts.")
            st.rerun()
        except (TranslationError, ValueError) as exc:
            st.error(f"Not replaced: {exc}")

    template = ctx.translations_template_path
    if template.exists():
        st.download_button("Download empty template", template.read_bytes(), file_name=template.name)

    with st.expander("Mock translations (for dry runs)"):
        st.caption("Writes [CS] + English. Refuses to overwrite a file with real translations.")
        if st.button("Create mock translations"):
            try:
                write_mock(RunContext(cfg))
                st.success("Mock translations written.")
                st.rerun()
            except TranslationError as exc:
                st.error(str(exc))

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
            st.dataframe(pass_rate_table(df).style.format("{:.1%}"), use_container_width=True)
        with right:
            st.markdown("**Status counts**")
            counts = df.groupby(["extraction_status", "exec_status"]).size().rename("rows").reset_index()
            st.dataframe(counts, hide_index=True, use_container_width=True)

        f1, f2, f3 = st.columns(3)
        models = f1.multiselect("Model", sorted(df["model"].unique()), default=sorted(df["model"].unique()))
        langs = f2.multiselect("Language", sorted(df["language"].unique()), default=sorted(df["language"].unique()))
        statuses = f3.multiselect("Exec status", sorted(df["exec_status"].unique()), default=sorted(df["exec_status"].unique()))
        view = df[df["model"].isin(models) & df["language"].isin(langs) & df["exec_status"].isin(statuses)]
        st.dataframe(view, hide_index=True, use_container_width=True, height=300)
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
