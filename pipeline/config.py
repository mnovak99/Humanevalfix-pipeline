"""Load config.yaml and .env into typed objects; resolve per-mode file paths."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"

API_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


@dataclass(frozen=True)
class Price:
    input: float
    output: float


@dataclass(frozen=True)
class ModelConfig:
    name: str
    provider: str
    enabled: bool
    max_tokens: int
    params: dict[str, Any] = field(default_factory=dict)
    price_per_mtok: Price = Price(0.0, 0.0)


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    config: str
    split: str
    revision: str | None


@dataclass(frozen=True)
class Config:
    dataset: DatasetConfig
    exec_workers: int
    api_workers: int
    retry_max_attempts: int
    cost_estimate_tokens: Price  # assumed tokens per call (input, output)
    instruction_template_en: str
    seed: int
    n_tasks: int
    pilot_n_tasks: int
    languages: list[str]
    runs_per_prompt: int
    exec_timeout_s: float
    translations_path: Path
    models: list[ModelConfig]

    @property
    def enabled_models(self) -> list[ModelConfig]:
        return [m for m in self.models if m.enabled]


def _parse_model(raw: dict[str, Any]) -> ModelConfig:
    price = raw.get("price_per_mtok") or {}
    return ModelConfig(
        name=str(raw["name"]),
        provider=str(raw["provider"]),
        enabled=bool(raw.get("enabled", True)),
        max_tokens=int(raw["max_tokens"]),
        params=dict(raw.get("params") or {}),
        price_per_mtok=Price(float(price.get("input", 0.0)), float(price.get("output", 0.0))),
    )


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    ds = raw["dataset"]
    cfg = Config(
        dataset=DatasetConfig(
            name=str(ds["name"]),
            config=str(ds["config"]),
            split=str(ds["split"]),
            revision=str(ds["revision"]) if ds.get("revision") else None,
        ),
        exec_workers=int(raw.get("exec_workers", 4)),
        api_workers=int(raw.get("api_workers", 4)),
        retry_max_attempts=int(raw.get("retry_max_attempts", 6)),
        cost_estimate_tokens=Price(
            float(raw["cost_estimate_tokens"]["input"]), float(raw["cost_estimate_tokens"]["output"])
        ),
        instruction_template_en=str(raw["instruction_template_en"]),
        seed=int(raw["seed"]),
        n_tasks=int(raw["n_tasks"]),
        pilot_n_tasks=int(raw["pilot_n_tasks"]),
        languages=[str(lang) for lang in raw["languages"]],
        runs_per_prompt=int(raw["runs_per_prompt"]),
        exec_timeout_s=float(raw["exec_timeout_s"]),
        translations_path=ROOT / raw["translations_path"],
        models=[_parse_model(m) for m in raw["models"]],
    )
    if cfg.pilot_n_tasks > cfg.n_tasks:
        raise ValueError("pilot_n_tasks must not exceed n_tasks (pilot must be a subset)")
    if "en" not in cfg.languages:
        raise ValueError("languages must include 'en' (the source language)")
    return cfg


def load_env() -> None:
    load_dotenv(ROOT / ".env", override=False)


def get_api_key(provider: str) -> str:
    """Return the key for a provider; never log the value."""
    var = API_KEY_ENV[provider]
    key = os.environ.get(var, "").strip()
    if not key:
        raise RuntimeError(f"{var} is not set (see .env.example)")
    return key


@dataclass(frozen=True)
class RunContext:
    """Run mode plus the file paths derived from it.

    --pilot and --dry-run add suffixes so test runs never overwrite real data.
    """

    config: Config
    pilot: bool = False
    dry_run: bool = False
    yes: bool = False
    root: Path = ROOT

    @property
    def n_tasks(self) -> int:
        return self.config.pilot_n_tasks if self.pilot else self.config.n_tasks

    def _suffix(self, include_dry: bool) -> str:
        s = "_pilot" if self.pilot else ""
        if include_dry and self.dry_run:
            s += "_dry"
        return s

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs"

    @property
    def tmp_dir(self) -> Path:
        return self.outputs_dir / "tmp"

    # Steps 01-02 do not depend on --dry-run (no model calls involved).
    @property
    def tasks_path(self) -> Path:
        return self.data_dir / f"tasks{self._suffix(False)}.json"

    @property
    def translations_template_path(self) -> Path:
        return self.data_dir / "translations_template.xlsx"

    @property
    def prompts_path(self) -> Path:
        return self.data_dir / f"prompts{self._suffix(False)}.jsonl"

    @property
    def responses_path(self) -> Path:
        return self.outputs_dir / f"responses{self._suffix(True)}.jsonl"

    @property
    def extracted_path(self) -> Path:
        return self.outputs_dir / f"extracted{self._suffix(True)}.jsonl"

    @property
    def executions_path(self) -> Path:
        return self.outputs_dir / f"executions{self._suffix(True)}.jsonl"

    @property
    def results_path(self) -> Path:
        return self.outputs_dir / f"results{self._suffix(True)}.csv"
