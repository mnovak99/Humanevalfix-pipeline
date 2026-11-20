"""Read/validate data/translations.xlsx and generate the mock version."""

from __future__ import annotations

from pipeline.config import RunContext

MOCK_PREFIX = "[CS] "


def write_mock(ctx: RunContext) -> None:
    raise NotImplementedError("mock-translations is not implemented yet (Phase 2)")
