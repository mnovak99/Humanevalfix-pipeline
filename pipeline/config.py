"""Load config.yaml and .env into typed objects; resolve per-mode file paths."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace
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
class Variant:
    """Named model-settings override for a follow-up run (`--variant NAME`)."""

    name: str
    description: str
    params: dict[str, dict[str, Any]]  # model name -> replacement `params`


@dataclass(frozen=True)
class Config:
    dataset_path: Path
    text_fields: dict[str, str]  # language -> dataset field with the task description
    prompt_templates: dict[str, str]  # language -> template with {field} placeholders
    exec_workers: int
    api_workers: int
    retry_max_attempts: int
    cost_estimate_tokens: Price  # assumed tokens per call (input, output)
    seed: int
    n_tasks: int
    pilot_n_tasks: int
    languages: list[str]
    runs_per_prompt: int
    exec_timeout_s: float
    models: list[ModelConfig]
    variants: dict[str, Variant] = field(default_factory=dict)

    @property
    def enabled_models(self) -> list[ModelConfig]:
        return [m for m in self.models if m.enabled]


class UnknownVariant(ValueError):
    pass


def apply_variant(cfg: Config, name: str | None) -> Config:
    """Config with the variant's `params` replacing those of the listed models."""
    if name is None:
        return cfg
    if name not in cfg.variants:
        raise UnknownVariant(f"unknown variant {name!r}; defined in config.yaml: {sorted(cfg.variants)}")
    variant = cfg.variants[name]
    models = [replace(m, params=dict(variant.params[m.name])) if m.name in variant.params else m for m in cfg.models]
    return replace(cfg, models=models)


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
    cfg = Config(
        dataset_path=ROOT / raw["dataset_path"],
        text_fields={str(k): str(v) for k, v in raw["text_fields"].items()},
        prompt_templates={str(k): str(v) for k, v in raw["prompt_templates"].items()},
        exec_workers=int(raw.get("exec_workers", 4)),
        api_workers=int(raw.get("api_workers", 4)),
        retry_max_attempts=int(raw.get("retry_max_attempts", 6)),
        cost_estimate_tokens=Price(
            float(raw["cost_estimate_tokens"]["input"]), float(raw["cost_estimate_tokens"]["output"])
        ),
        seed=int(raw["seed"]),
        n_tasks=int(raw["n_tasks"]),
        pilot_n_tasks=int(raw["pilot_n_tasks"]),
        languages=[str(lang) for lang in raw["languages"]],
        runs_per_prompt=int(raw["runs_per_prompt"]),
        exec_timeout_s=float(raw["exec_timeout_s"]),
        models=[_parse_model(m) for m in raw["models"]],
        variants={
            str(name): Variant(
                name=str(name),
                description=str(v.get("description", "")),
                params={str(model): dict(spec.get("params") or {}) for model, spec in (v.get("models") or {}).items()},
            )
            for name, v in (raw.get("variants") or {}).items()
        },
    )
    model_names = {m.name for m in cfg.models}
    for v in cfg.variants.values():
        if not re.fullmatch(r"[a-z0-9]+", v.name):
            raise ValueError(f"variant name {v.name!r} must be lowercase letters/digits (used in file names)")
        unknown = set(v.params) - model_names
        if unknown:
            raise ValueError(f"variant {v.name!r} refers to unknown models {sorted(unknown)}")
    if cfg.pilot_n_tasks > cfg.n_tasks:
        raise ValueError("pilot_n_tasks must not exceed n_tasks (pilot must be a subset)")
    if "en" not in cfg.languages:
        raise ValueError("languages must include 'en' (the source language)")
    for section in ("text_fields", "prompt_templates"):
        missing = [lang for lang in cfg.languages if lang not in getattr(cfg, section)]
        if missing:
            raise ValueError(f"{section} has no entry for languages {missing}")
    return cfg


def load_env() -> None:
    load_dotenv(ROOT / ".env", override=False)


class MissingApiKey(RuntimeError):
    pass


def get_api_key(provider: str) -> str:
    """Return the key for a provider; never log the value."""
    var = API_KEY_ENV[provider]
    key = os.environ.get(var, "").strip()
    if not key:
        raise MissingApiKey(f"{var} is not set (copy .env.example to .env and fill it in)")
    return key


@dataclass(frozen=True)
class RunContext:
    """Run mode plus the file paths derived from it.

    --pilot, --dry-run, smoke mode and --variant add suffixes so test runs never overwrite
    real data. A variant changes only model settings, so it suffixes the output files but
    shares tasks and prompts (e.g. responses_pilot_noreason.jsonl, prompts_pilot.jsonl).
    `config` must already have the variant applied (see apply_variant).
    """

    config: Config
    pilot: bool = False
    dry_run: bool = False
    yes: bool = False
    smoke: bool = False  # EN-only API check; reads the full task set
    variant: str | None = None
    root: Path = ROOT

    @property
    def n_tasks(self) -> int:
        return self.config.pilot_n_tasks if self.pilot else self.config.n_tasks

    def _suffix(self, outputs: bool) -> str:
        s = "_pilot" if self.pilot else ""
        if self.smoke:
            s += "_smoke"
        if outputs and self.variant:
            s += f"_{self.variant}"
        if outputs and self.dry_run:
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
        return self.data_dir / f"tasks{'_pilot' if self.pilot else ''}.json"

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
