"""Check each translation against the English text (used by step 02 and the web UI).

Errors (step 02 fails): doctest lines and their expected output must be identical, the
entry point must be named, and backticked identifiers must be kept.
Warnings: length ratio outside LENGTH_RATIO_RANGE, numbers from English missing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

LENGTH_RATIO_RANGE = (0.6, 1.8)

DOCTEST = "doctest lines differ"
ENTRY_POINT = "entry point missing"
BACKTICKS = "backticked identifier missing"
LENGTH = "length ratio"
NUMBERS = "numbers missing"
ERROR_CHECKS = (DOCTEST, ENTRY_POINT, BACKTICKS)
WARNING_CHECKS = (LENGTH, NUMBERS)

_BACKTICK_RE = re.compile(r"`([^`\n]+)`")
_NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.]*\w)")


@dataclass(frozen=True)
class Issue:
    check: str
    task_id: str
    language: str
    detail: str


@dataclass
class ValidationReport:
    errors: list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def ids(self, check: str) -> list[str]:
        issues = self.errors if check in ERROR_CHECKS else self.warnings
        return sorted({i.task_id for i in issues if i.check == check}, key=lambda t: int(t.split("/")[1]))

    def summary(self) -> str:
        lines = []
        for check in (*ERROR_CHECKS, *WARNING_CHECKS):
            ids = self.ids(check)
            kind = "ERROR" if check in ERROR_CHECKS else "warning"
            lines.append(f"{kind:7} {check}: {len(ids)}" + (f" {ids}" if ids else ""))
        return "\n".join(lines)


def doctest_pairs(text: str) -> list[tuple[str, str]]:
    """Each `>>>` line with the line right after it (its expected output), both stripped."""
    lines = text.split("\n")
    return [
        (line.strip(), lines[i + 1].strip() if i + 1 < len(lines) else "")
        for i, line in enumerate(lines)
        if line.lstrip().startswith(">>>")
    ]


def numbers(text: str) -> set[str]:
    return set(_NUMBER_RE.findall(text))


def check_task(task: dict, language: str, en_field: str, tr_field: str) -> tuple[list[Issue], list[Issue]]:
    tid, en, tr = task["task_id"], task[en_field], task[tr_field]
    errors: list[Issue] = []
    warnings: list[Issue] = []

    en_pairs, tr_pairs = doctest_pairs(en), doctest_pairs(tr)
    if en_pairs != tr_pairs:
        diff = next(
            (f"EN {a!r} vs {language.upper()} {b!r}" for a, b in zip(en_pairs, tr_pairs) if a != b),
            f"{len(en_pairs)} doctest lines in EN, {len(tr_pairs)} in {language.upper()}",
        )
        errors.append(Issue(DOCTEST, tid, language, diff))
    if task["entry_point"] not in tr:
        errors.append(Issue(ENTRY_POINT, tid, language, task["entry_point"]))
    lost = [ident for ident in dict.fromkeys(_BACKTICK_RE.findall(en)) if ident not in tr]
    if lost:
        errors.append(Issue(BACKTICKS, tid, language, ", ".join(lost)))

    ratio = len(tr) / max(1, len(en))
    lo, hi = LENGTH_RATIO_RANGE
    if not lo <= ratio <= hi:
        warnings.append(Issue(LENGTH, tid, language, f"{ratio:.2f}"))
    missing = sorted(numbers(en) - numbers(tr), key=lambda s: (len(s), s))
    if missing:
        warnings.append(Issue(NUMBERS, tid, language, ", ".join(missing[:8])))
    return errors, warnings


def validate(tasks: list[dict], text_fields: dict[str, str]) -> ValidationReport:
    report = ValidationReport()
    en_field = text_fields["en"]
    for lang, tr_field in text_fields.items():
        if lang == "en":
            continue
        for task in tasks:
            errors, warnings = check_task(task, lang, en_field, tr_field)
            report.errors += errors
            report.warnings += warnings
    return report
