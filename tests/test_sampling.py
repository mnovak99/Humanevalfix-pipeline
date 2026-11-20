from collections import Counter

from pipeline.s01_prepare import preamble_of, stratified_order

STRATA = {
    "value misuse": [f"V{i}" for i in range(44)],
    "missing logic": [f"M{i}" for i in range(33)],
    "excess logic": [f"E{i}" for i in range(31)],
    "operator misuse": [f"O{i}" for i in range(25)],
    "variable misuse": [f"A{i}" for i in range(23)],
    "function misuse": [f"F{i}" for i in range(8)],
}


def _stratum(tid: str) -> str:
    return next(k for k, v in STRATA.items() if tid in v)


def test_deterministic_and_proportional() -> None:
    a = stratified_order(STRATA, 100, 42)
    assert a == stratified_order(STRATA, 100, 42)
    assert a != stratified_order(STRATA, 100, 43)
    assert len(a) == len(set(a)) == 100
    counts = Counter(_stratum(t) for t in a)
    total = sum(len(v) for v in STRATA.values())
    for k, v in STRATA.items():
        assert abs(counts[k] - 100 * len(v) / total) < 1


def test_prefix_is_roughly_stratified() -> None:
    pilot = stratified_order(STRATA, 100, 42)[:20]
    counts = Counter(_stratum(t) for t in pilot)
    total = sum(len(v) for v in STRATA.values())
    for k, v in STRATA.items():
        assert abs(counts[k] - 20 * len(v) / total) <= 1.5


def test_preamble_keeps_imports_and_helpers() -> None:
    prompt = 'from typing import List\n\n\ndef helper(x):\n    return x\n\n\ndef main(xs: List[int]):\n    """Doc."""\n'
    pre = preamble_of(prompt, "main")
    assert "from typing import List" in pre and "def helper" in pre and "def main" not in pre
