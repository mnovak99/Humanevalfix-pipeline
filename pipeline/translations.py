"""Translation file contract (data/translations.xlsx): write template/mock, load and validate.

Sheet `docstrings`: task_id | docstring_<lang>... | translated_by | reviewed_by | status | note
Sheet `instructions`: language | instruction   (may use the {entry_point} placeholder)
"""

from __future__ import annotations

import logging
import re
import string
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from pipeline.config import Config, RunContext
from pipeline.docstrings import MARKER_RE, check_markers
from pipeline.io import read_json

log = logging.getLogger(__name__)

STATUSES = ("todo", "translated", "reviewed")
META_COLUMNS = ["translated_by", "reviewed_by", "status", "note"]
LENGTH_RATIO_RANGE = (0.6, 1.8)
ALLOWED_PLACEHOLDERS = {"entry_point"}

TRANSLATOR_GUIDE = [
    "Translate docstring_en into docstring_<lang>; keep line breaks roughly as in English.",
    "Lines {EXAMPLE_1}, {EXAMPLE_2}, ... are code examples inserted automatically: keep each on its own line, in the same order, unchanged.",
    "Values inside prose (e.g. 'Input: [4,2,3]', 'For num = \"AB\" the output should be 1') must be copied exactly; translate only the words.",
    "Do not translate identifiers (function/parameter names) and do not use triple quotes (\"\"\" or ''').",
    "Status: todo -> translated -> reviewed. Only 'reviewed' rows can be used in a live run.",
    "Sheet 'instructions': one row per language; keep the {entry_point} placeholder.",
    "Rows are in sampling order: the first pilot_n_tasks rows form the pilot set.",
]

# Literal-looking values in prose that a translator must copy verbatim (checked as warnings).
_LITERAL_RE = re.compile(r"\"[^\"\n]*\"|'[^'\n]*'|\[[^\[\]\n]*\]|\(-?\d[^()\n]*\)|-?\d+(?:\.\d+)?")


def mock_prefix(lang: str) -> str:
    return f"[{lang.upper()}] "


def is_mock_text(text: str, lang: str) -> bool:
    return lang != "en" and text.startswith(mock_prefix(lang))


def doc_col(lang: str) -> str:
    return f"docstring_{lang}"


class TranslationError(ValueError):
    pass


# --------------------------------------------------------------------------- writing


def write_workbook(path: Path, tasks: list[dict], cfg: Config, mock: bool = False) -> None:
    """Template (other languages empty, status todo) or mock ("[CS] " + EN, status reviewed)."""
    others = [lang for lang in cfg.languages if lang != "en"]
    rows = []
    for t in tasks:
        en = t["docstring_parts"]["description"]
        row = {"task_id": t["task_id"], doc_col("en"): en}
        for lang in others:
            # Keep a leading marker on its own line so the mock stays valid.
            row[doc_col(lang)] = (mock_prefix(lang) + ("\n" if MARKER_RE.match(en.split("\n")[0]) else "") + en) if mock else ""
        row |= {
            "translated_by": "mock" if mock else "",
            "reviewed_by": "mock" if mock else "",
            "status": "reviewed" if mock else "todo",
            "note": "",
        }
        rows.append(row)
    columns = ["task_id", *[doc_col(lang) for lang in cfg.languages], *META_COLUMNS]
    instructions = []
    for lang in cfg.languages:
        text = cfg.instruction_template_en if lang == "en" else (mock_prefix(lang) + cfg.instruction_template_en if mock else "")
        instructions.append({"language": lang, "instruction": text})

    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        pd.DataFrame(rows, columns=columns).to_excel(xw, sheet_name="docstrings", index=False)
        pd.DataFrame(instructions, columns=["language", "instruction"]).to_excel(xw, sheet_name="instructions", index=False)
        pd.DataFrame({"guide": TRANSLATOR_GUIDE}).to_excel(xw, sheet_name="guide", index=False)
        ws = xw.sheets["docstrings"]
        widths = [12] + [80] * len(cfg.languages) + [14, 14, 12, 30]
        for i, width in enumerate(widths):
            ws.column_dimensions[chr(ord("A") + i)].width = width
        xw.sheets["instructions"].column_dimensions["B"].width = 60
        xw.sheets["guide"].column_dimensions["A"].width = 140


def write_mock(ctx: RunContext, force: bool = False) -> Path:
    """Create a mock translations file from the full task set (data/tasks.json)."""
    full_tasks_path = RunContext(ctx.config, root=ctx.root).tasks_path
    if not full_tasks_path.exists():
        raise TranslationError(f"{full_tasks_path} not found; run step 01 first")
    path = ctx.config.translations_path
    if path.exists() and not force and not is_mock_file(path, ctx.config.languages):
        raise TranslationError(f"{path} contains non-mock translations; refusing to overwrite (use --force)")
    tasks = read_json(full_tasks_path)["tasks"]
    write_workbook(path, tasks, ctx.config, mock=True)
    log.info("wrote mock translations for %d tasks to %s", len(tasks), path)
    return path


def is_mock_file(path: Path, languages: list[str]) -> bool:
    try:
        sheets = _read_sheets(path)
    except Exception:
        return False
    df = sheets["docstrings"]
    others = [lang for lang in languages if lang != "en" and doc_col(lang) in df.columns]
    return bool(others) and all(
        df[doc_col(lang)].map(lambda s, lang=lang: s == "" or is_mock_text(s, lang)).all() for lang in others
    )


# --------------------------------------------------------------------------- reading


@dataclass
class Translations:
    docstrings: dict[str, dict[str, str]]  # lang -> task_id -> description
    instructions: dict[str, str]  # lang -> instruction template
    mock_languages: set[str] = field(default_factory=set)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_mock(self) -> bool:
        return bool(self.mock_languages)


def _normalize(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _read_sheets(path: Path) -> dict[str, pd.DataFrame]:
    return pd.read_excel(path, sheet_name=None, dtype=str, keep_default_na=False)


def _placeholders(template: str) -> set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(template) if name is not None}


def _literals(text: str) -> list[str]:
    prose = "\n".join(ln for ln in text.split("\n") if not MARKER_RE.match(ln))
    return _LITERAL_RE.findall(prose)


def _prose_len(text: str) -> int:
    return len("".join(ln for ln in text.split("\n") if not MARKER_RE.match(ln)).strip())


def load_and_validate(
    path: Path,
    languages: list[str],
    tasks: list[dict],
    known_task_ids: set[str],
) -> Translations:
    """Validate the translation file for `tasks`; raise TranslationError listing all problems.

    Rows for `tasks` must exist, be non-empty and `reviewed`. Rows for other task ids are
    allowed only if they are in `known_task_ids` (the pilot uses a subset of the full file).
    """
    if not path.exists():
        raise TranslationError(f"{path} not found (create it with `python -m pipeline mock-translations`)")
    sheets = _read_sheets(path)
    errors: list[str] = []
    warnings: list[str] = []
    for sheet in ("docstrings", "instructions"):
        if sheet not in sheets:
            raise TranslationError(f"{path}: missing sheet {sheet!r}")
    df = sheets["docstrings"]
    df.columns = [str(c).strip() for c in df.columns]
    required = ["task_id", *[doc_col(lang) for lang in languages], "status"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise TranslationError(f"{path}: docstrings sheet is missing columns {missing_cols}")
    df["task_id"] = df["task_id"].str.strip()
    df["status"] = df["status"].str.strip().str.lower()

    needed = [t["task_id"] for t in tasks]
    present = df["task_id"].tolist()
    dupes = sorted({tid for tid in present if present.count(tid) > 1})
    missing = [tid for tid in needed if tid not in set(present)]
    extras = sorted(set(present) - known_task_ids - set(needed))
    if dupes:
        errors.append(f"duplicate task_ids: {dupes}")
    if missing:
        errors.append(f"missing task_ids: {missing}")
    if extras:
        errors.append(f"unknown task_ids (not in the sampled task set): {extras}")

    rows = df.drop_duplicates("task_id").set_index("task_id")
    used = [tid for tid in needed if tid in rows.index]
    bad_status = [tid for tid in used if rows.at[tid, "status"] != "reviewed"]
    if bad_status:
        errors.append(f"status is not 'reviewed': {bad_status}")
    invalid_status = sorted({s for s in rows["status"] if s not in STATUSES})
    if invalid_status:
        errors.append(f"invalid status values {invalid_status} (allowed: {list(STATUSES)})")

    by_id = {t["task_id"]: t for t in tasks}
    docstrings: dict[str, dict[str, str]] = {lang: {} for lang in languages}
    mock_languages: set[str] = set()
    for lang in languages:
        empty, marker_bad, quote_bad = [], [], []
        for tid in used:
            text = _normalize(rows.at[tid, doc_col(lang)])
            parts = by_id[tid]["docstring_parts"]
            if not text.strip():
                empty.append(tid)
                continue
            docstrings[lang][tid] = text
            if lang == "en":
                if text != parts["description"]:
                    errors.append(f"{tid}: docstring_en differs from data/tasks.json (re-export the template)")
                continue
            if is_mock_text(text, lang):
                mock_languages.add(lang)
            problems = check_markers(text, len(parts["examples"]))
            if problems:
                marker_bad.append(f"{tid} ({'; '.join(problems)})")
            if '"""' in text or "'''" in text:
                quote_bad.append(tid)
            if "\\" in text and "\\" not in parts["description"]:
                warnings.append(f"{tid} [{lang}]: contains a backslash (escape sequences change the docstring)")
            ratio = _prose_len(text) / max(1, _prose_len(parts["description"]))
            lo, hi = LENGTH_RATIO_RANGE
            if not lo <= ratio <= hi:
                warnings.append(f"{tid} [{lang}]: length ratio {ratio:.2f} outside {lo}-{hi}")
            lost = [lit for lit in dict.fromkeys(_literals(parts["description"])) if lit not in text]
            if lost:
                warnings.append(f"{tid} [{lang}]: values from EN not found verbatim: {lost[:5]}")
        if empty:
            errors.append(f"empty {doc_col(lang)}: {empty}")
        if marker_bad:
            errors.append(f"bad example markers in {doc_col(lang)}: {marker_bad}")
        if quote_bad:
            errors.append(f"triple quotes in {doc_col(lang)}: {quote_bad}")

    inst_df = sheets["instructions"]
    inst_df.columns = [str(c).strip() for c in inst_df.columns]
    instructions: dict[str, str] = {}
    for _, r in inst_df.iterrows():
        instructions[str(r["language"]).strip()] = _normalize(str(r["instruction"])).strip()
    for lang in languages:
        text = instructions.get(lang, "")
        if not text:
            errors.append(f"instruction for {lang!r} is missing or empty")
            continue
        try:
            fields = _placeholders(text)
        except ValueError as exc:
            errors.append(f"instruction for {lang!r} is not a valid template: {exc}")
            continue
        if fields - ALLOWED_PLACEHOLDERS:
            errors.append(f"instruction for {lang!r} uses unknown placeholders {sorted(fields - ALLOWED_PLACEHOLDERS)}")
        if lang != "en" and "en" in instructions and fields != _placeholders(instructions["en"]):
            errors.append(f"instruction for {lang!r} has placeholders {sorted(fields)}, EN has {sorted(_placeholders(instructions['en']))}")
        if is_mock_text(text, lang):
            mock_languages.add(lang)

    if errors:
        raise TranslationError(f"{path}: {len(errors)} problem(s):\n  - " + "\n  - ".join(errors))
    return Translations(docstrings, {lang: instructions[lang] for lang in languages}, mock_languages, warnings)
