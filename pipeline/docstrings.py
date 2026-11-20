"""Split a function docstring into translatable description and verbatim examples, and rebuild it.

The docstring content (text between the quotes) is split into lines and each line is
classified as example or prose. Each maximal run of example lines becomes a verbatim
block and is replaced in the description by a marker line ``{EXAMPLE_k}``. Translators
translate the description and keep the markers; rebuilding re-inserts the blocks, so
examples stay byte-identical across languages by construction.

Layout (the leading newline after the opening quotes, the common indentation and the
closing-quote line) is structural and not part of the description, so it is identical
in every language. Rebuilding with the original English description reproduces the
original source exactly.
"""

from __future__ import annotations

import ast
import os
import re
from dataclasses import asdict, dataclass

MARKER_RE = re.compile(r"^\s*\{EXAMPLE_(\d+)\}\s*$")

# A line that starts with a call, optionally as a bullet or with a lead-in word:
# "* f(x) => y", "assert f(x) => y", "for f(x) == y".
_CALL_RE = re.compile(r"^\s*(?:[*\-]\s+|assert\s+|for\s+)?([A-Za-z_]\w*)\s*\(")
# Something that maps a call to its expected result.
_RESULT_RE = re.compile(r"==|=>|➞|->|→|\breturns?\b|\bshould return\b|\s=\s|#")
_QUOTES = ('"""', "'''")


def marker(k: int) -> str:
    return f"{{EXAMPLE_{k}}}"


@dataclass(frozen=True)
class DocstringParts:
    start: int  # offset of the docstring content (just after the opening quotes)
    end: int  # offset of the closing quotes
    head: str  # whitespace after the opening quotes when the text starts on the next line
    has_head: bool
    tail: str  # whitespace before the closing quotes if they sit on their own line
    has_tail: bool
    indent: str  # common indentation of continuation lines
    description: str  # translatable text, examples replaced by {EXAMPLE_k} lines
    examples: list[str]  # verbatim example blocks, k-th block = examples[k-1]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> DocstringParts:
        return cls(**d)


class DocstringError(ValueError):
    pass


def _offset(line_starts: list[int], source: str, lineno: int, col_bytes: int) -> int:
    """Convert ast (1-based line, UTF-8 byte column) to a character offset."""
    start = line_starts[lineno - 1]
    line_end = source.find("\n", start)
    line = source[start : line_end if line_end != -1 else len(source)]
    return start + len(line.encode("utf-8")[:col_bytes].decode("utf-8"))


def locate_docstring(source: str, func_name: str) -> tuple[int, int]:
    """Return (content_start, content_end) offsets of the top-level function's docstring."""
    tree = ast.parse(source)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == func_name]
    if len(funcs) != 1:
        raise DocstringError(f"expected exactly one top-level def {func_name!r}, found {len(funcs)}")
    # Usually body[0]; HumanEvalFix Python/115 has `import math` before the string literal.
    strings = [
        s.value
        for s in funcs[0].body
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str)
    ]
    if not strings:
        raise DocstringError(f"{func_name!r} has no docstring")
    node = strings[0]
    assert node.end_lineno is not None and node.end_col_offset is not None
    line_starts = [0]
    for i, ch in enumerate(source):
        if ch == "\n":
            line_starts.append(i + 1)
    tok_start = _offset(line_starts, source, node.lineno, node.col_offset)
    tok_end = _offset(line_starts, source, node.end_lineno, node.end_col_offset)
    token = source[tok_start:tok_end]
    body = token.lstrip("rRuU")
    quote = body[:3]
    if quote not in _QUOTES or not body.endswith(quote):
        raise DocstringError(f"{func_name!r}: docstring is not a triple-quoted literal")
    prefix_len = len(token) - len(body)
    return tok_start + prefix_len + 3, tok_end - 3


def _bracket_depth(text: str) -> int:
    depth = 0
    quote: str | None = None
    for ch in text:
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
    return depth


def classify_lines(lines: list[str], func_names: set[str]) -> list[bool]:
    """True for lines that belong to an example (kept verbatim)."""
    is_ex = [False] * len(lines)
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith(">>>"):
            is_ex[i] = True
            i += 1
            # Expected output: following non-blank lines up to the next blank or prompt.
            while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith(">>>"):
                is_ex[i] = True
                i += 1
            continue
        m = _CALL_RE.match(lines[i])
        if m and (m.group(1) in func_names or _RESULT_RE.search(lines[i][m.end() :])):
            # Call example, possibly spanning several lines until brackets balance.
            depth = _bracket_depth(lines[i])
            is_ex[i] = True
            i += 1
            while depth > 0 and i < len(lines):
                depth += _bracket_depth(lines[i])
                is_ex[i] = True
                i += 1
            continue
        i += 1
    return is_ex


def _common_indent(lines: list[str]) -> str:
    leads = [ln[: len(ln) - len(ln.lstrip())] for ln in lines if ln.strip()]
    return os.path.commonprefix(leads) if leads else ""


def split_docstring(source: str, func_name: str, func_names: set[str] | None = None) -> DocstringParts:
    start, end = locate_docstring(source, func_name)
    lines = source[start:end].split("\n")
    names = (func_names or set()) | {func_name}

    has_head = len(lines) > 1 and lines[0].strip() == ""
    head = lines[0] if has_head else ""
    has_tail = len(lines) > 1 and lines[-1].strip() == ""
    tail = lines[-1] if has_tail else ""
    body = lines[int(has_head) : len(lines) - int(has_tail)]
    first_is_line0 = not has_head

    cont = body[1:] if first_is_line0 else body
    indent = _common_indent(cont)
    is_ex = classify_lines(body, names)

    desc: list[str] = []
    examples: list[str] = []
    i = 0
    while i < len(body):
        if is_ex[i]:
            j = i
            while j < len(body) and is_ex[j]:
                j += 1
            examples.append("\n".join(body[i:j]))
            desc.append(marker(len(examples)))
            i = j
            continue
        line = body[i]
        if (i == 0 and first_is_line0) or not line.strip():
            desc.append(line)
        else:
            desc.append(line[len(indent) :])
        i += 1

    description = "\n".join(desc)
    if any(MARKER_RE.match(ln) for ln, ex in zip(body, is_ex) if not ex):
        raise DocstringError(f"{func_name!r}: docstring already contains a marker-like line")

    parts = DocstringParts(
        start=start,
        end=end,
        head=head,
        has_head=has_head,
        tail=tail,
        has_tail=has_tail,
        indent=indent,
        description=description,
        examples=examples,
    )
    rebuilt = rebuild_source(source, parts, description)
    if rebuilt != source:
        raise DocstringError(f"{func_name!r}: docstring split does not round-trip")
    return parts


def check_markers(description: str, n_examples: int) -> list[str]:
    """Return problems with the example markers in a (translated) description."""
    found = [int(m.group(1)) for ln in description.split("\n") if (m := MARKER_RE.match(ln))]
    expected = list(range(1, n_examples + 1))
    if found != expected:
        return [f"example markers {found} != expected {expected}"]
    stray = [ln for ln in description.split("\n") if "{EXAMPLE_" in ln and not MARKER_RE.match(ln)]
    return [f"marker not on its own line: {ln!r}" for ln in stray]


def rebuild_content(parts: DocstringParts, description: str) -> str:
    problems = check_markers(description, len(parts.examples))
    if problems:
        raise DocstringError("; ".join(problems))
    out: list[str] = [parts.head] if parts.has_head else []
    for j, line in enumerate(description.split("\n")):
        m = MARKER_RE.match(line)
        if m:
            out.append(parts.examples[int(m.group(1)) - 1])
        elif (j == 0 and not parts.has_head) or not line.strip():
            out.append(line)
        else:
            out.append(parts.indent + line)
    if parts.has_tail:
        out.append(parts.tail)
    return "\n".join(out)


def rebuild_source(source: str, parts: DocstringParts, description: str) -> str:
    """Replace the docstring content of `source` using `description` (any language)."""
    return source[: parts.start] + rebuild_content(parts, description) + source[parts.end :]


def strip_docstring(source: str, parts: DocstringParts) -> str:
    """Source with the docstring content removed (used for the cross-language invariant)."""
    return source[: parts.start] + source[parts.end :]
