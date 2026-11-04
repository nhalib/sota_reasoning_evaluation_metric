#!/usr/bin/env python3
"""Execution core for Empirical Probes (Section 6).

Runs Probe 1 (Uncertainty Coherence) and Probe 2 (Monotonicity Defeasibility)
against live reasoning model endpoints.

This module is a library only: it exposes run_suite() and the setup helpers the
master controller (run_master.py) reuses, and has no command-line interface of
its own. run_master.py is the single entry point for the suite, including
single-model runs (one --model flag).
"""

import sys
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Add project root (probes/) to path so the suite runs from within probes/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.models import (
    BayesianHarnessModel,
    OpenAICompatibleModel
)
from probe1_uncertainty.runner import evaluate_model_probe1
from probe2_monotonicity.runner import evaluate_model_probe2

from data_loaders import load_standard_gsm8k, load_standard_prontoqa
from probe1_uncertainty.dataset import PROBE1_DATASET
from probe2_monotonicity.dataset import PROBE2_DATASET


@dataclass
class SuiteConfig:
    mode: str = "all"                       # all | probe1 | probe2
    dataset: str = "standard"               # standard | seed
    n_items: int = 20
    include_scaffold: bool = True
    api_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 384

    def validate(self):
        if self.mode not in ("all", "probe1", "probe2"):
            raise ValueError(f"Invalid mode: {self.mode!r}")
        if self.dataset not in ("standard", "seed"):
            raise ValueError(f"Invalid dataset: {self.dataset!r}")


# Env-var names for the API key, consulted by endpoint host so a multi-provider
# sweep picks the right credential per model without leaking one key to all.
PROVIDER_KEY_ENVS = {
    "deepinfra": ("DEEPINFRA_API_KEY",),
    "together": ("TOGETHER_API_KEY",),
    "opencode": ("OPENCODE_API_KEY",),
}


def _resolve_api_key(api_url: Optional[str]) -> str:
    host = (api_url or "").lower()
    for token, names in PROVIDER_KEY_ENVS.items():
        if token in host:
            for name in names:
                val = os.getenv(name)
                if val:
                    return val
    for name in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY"):
        val = os.getenv(name)
        if val:
            return val
    return "EMPTY"


def resolve_endpoint(cfg: SuiteConfig) -> SuiteConfig:
    """Resolves credentials from environment/.env based on the configured api_url."""
    if not cfg.api_url:
        models_file = Path("models.json")
        if models_file.exists():
            try:
                import json
                raw = json.loads(models_file.read_text(encoding="utf-8"))
                if isinstance(raw, dict) and "default_api_url" in raw:
                    cfg.api_url = raw["default_api_url"]
            except Exception:
                pass
    if not cfg.api_key:
        cfg.api_key = _resolve_api_key(cfg.api_url)
    return cfg


def load_probe_datasets(cfg: SuiteConfig) -> tuple:
    """Loads the P1/P2 item sets once per sweep. Returns (p1_items, p2_items)."""
    if cfg.dataset == "standard":
        print(f"[*] Loading Standard Benchmarks ({cfg.n_items} items each: GSM8K for P1, ProntoQA for P2)...")
        p1_dataset = load_standard_gsm8k(n_items=cfg.n_items) if cfg.mode in ("all", "probe1") else []
        p2_dataset = load_standard_prontoqa(n_items=cfg.n_items) if cfg.mode in ("all", "probe2") else []
    else:
        print("[*] Loading Handcrafted Seed Datasets...")
        p1_dataset = PROBE1_DATASET
        p2_dataset = PROBE2_DATASET
    return p1_dataset, p2_dataset


def build_models(cfg: SuiteConfig, model_id: str) -> list:
    """Instantiates the target model (plus Bayesian harness scaffold if enabled)."""
    api_model = OpenAICompatibleModel(
        name=model_id,
        base_url=cfg.api_url,
        api_key=cfg.api_key,
        model_id=model_id,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )
    models = [api_model]
    if cfg.include_scaffold:
        models.append(BayesianHarnessModel(api_model))
    return models


def evaluate_model(model, cfg: SuiteConfig, p1_dataset: List, p2_dataset: List) -> tuple:
    """Runs the probe suite on a single model. Returns (p1_out, p2_out)."""
    print(f"\n--- Evaluating Model: {model.name} ---")
    p1_out = p2_out = None
    if cfg.mode in ("all", "probe1"):
        p1_out = evaluate_model_probe1(model, p1_dataset)
        print(f"    [Probe 1] Verbalized Brier: {p1_out['verbalized_brier']} | Logprob Brier: {p1_out['logprob_brier']} | Additivity Viol: {p1_out['additivity_violation_rate']}%")
    if cfg.mode in ("all", "probe2") and not getattr(model, "base_model", None):
        p2_out = evaluate_model_probe2(model, p2_dataset)
        rf = p2_out['flip_rates_by_condition']
        print(f"    [Probe 2] Flip Rates: Conjunct={rf['conjunct']}% | Inst={rf['instantiation']}% | Para={rf['paraphrase']}% (Overall={p2_out['overall_flip_rate']}%)")
    return p1_out, p2_out


def run_suite(
    model_ids: Sequence[str],
    cfg: SuiteConfig,
    datasets: Optional[tuple] = None,
    on_model_error=None,
    **kwargs,
) -> Dict[str, Any]:
    """Core evaluation routine.

    Loads datasets once, then evaluates every model in-process. Partial results
    are preserved: a failing model is recorded under 'errors' (via
    on_model_error(name, exc) if given) and the sweep continues.
    """
    cfg.validate()
    if not cfg.api_url or not cfg.api_key:
        raise ValueError(
            "No model endpoint configured: set api_url/api_key on SuiteConfig "
            "or API_BASE_URL/OPENCODE_API_KEY (or equivalents) in the environment or .env."
        )

    p1_dataset, p2_dataset = datasets if datasets is not None else load_probe_datasets(cfg)

    all_p1_results: Dict[str, Any] = {}
    all_p2_results: Dict[str, Any] = {}
    errors: Dict[str, str] = {}

    for model_id in model_ids:
        models = build_models(cfg, model_id)
        for model in models:
            try:
                p1_out, p2_out = evaluate_model(model, cfg, p1_dataset, p2_dataset)
            except Exception as exc:
                print(f"    [ERROR] {model.name}: {type(exc).__name__}: {exc}")
                errors[model.name] = f"{type(exc).__name__}: {exc}"
                if on_model_error:
                    on_model_error(model.name, exc)
                continue
            if p1_out is not None:
                all_p1_results[model.name] = p1_out
            if p2_out is not None:
                all_p2_results[model.name] = p2_out

    return {
        "run_config": {
            "mode": cfg.mode,
            "dataset": cfg.dataset,
            "n_items": cfg.n_items,
            "temperature": cfg.temperature,
            "include_scaffold": cfg.include_scaffold,
            "api_url": cfg.api_url,
            "model_ids": list(model_ids),
        },
        "probe1_models": all_p1_results,
        "probe2_models": all_p2_results,
        "errors": errors,
    }
