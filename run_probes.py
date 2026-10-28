#!/usr/bin/env python3
"""Execution core for Empirical Probes (Section 6).

Runs Probe 1 (Uncertainty Coherence) and Probe 2 (Monotonicity Defeasibility)
against classical positive controls and live reasoning model endpoints.
All evaluation logic lives in run_suite(), which is importable so the master
controller can reuse loaded datasets and classical-control results across
models. main() wraps it for single-model CLI runs, writing the raw evaluation
output to a JSON document.
"""

import sys
import os
import json
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Add project root (probes/) to path so the suite runs from within probes/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.config import RESULTS_DIR
from common.models import (
    BayesianHarnessModel,
    OpenAICompatibleModel
)
from probe1_uncertainty.problog_control import run_problog_control
from probe1_uncertainty.runner import evaluate_model_probe1
from probe2_monotonicity.clingo_control import run_clingo_control
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


def resolve_endpoint(cfg: SuiteConfig) -> SuiteConfig:
    """Fills missing endpoint config from environment variables / .env."""
    if not cfg.api_url:
        cfg.api_url = os.getenv("OPENCODE_BASE_URI") or os.getenv("OPENAI_BASE_URL") or os.getenv("API_BASE_URL")
    if not cfg.api_key:
        cfg.api_key = os.getenv("OPENCODE_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or "EMPTY"
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


def run_classical_controls(p1_dataset: List, p2_dataset: List, mode: str) -> tuple:
    """Runs ProbLog/Clingo controls (model-independent). Returns (problog_res, clingo_res)."""
    problog_res = clingo_res = None
    if mode in ("all", "probe1"):
        print("[*] Executing Probe 1 Classical Control: ProbLog Exact Solver...")
        problog_res = run_problog_control(p1_dataset)
        print(f"    -> ProbLog Additivity Violation Rate: {problog_res['additivity_violation_rate']*100:.1f}% (Coherent by construction)")
    if mode in ("all", "probe2"):
        print("[*] Executing Probe 2 Classical Control: Clingo (ASP Deductive Fragment)...")
        clingo_res = run_clingo_control(p2_dataset)
        print(f"    -> Clingo Flip Rate: {clingo_res['overall_flip_rate']:.1f}% (Monotone by construction)")
    return problog_res, clingo_res


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
    controls: Optional[tuple] = None,
    datasets: Optional[tuple] = None,
    on_model_error=None,
) -> Dict[str, Any]:
    """Core evaluation routine.

    Loads datasets and runs classical controls once, then evaluates every
    model in-process. Partial results are preserved: a failing model is
    recorded under 'errors' (via on_model_error(name, exc) if given) and the
    sweep continues.
    """
    cfg.validate()
    if not cfg.api_url or not cfg.api_key:
        raise ValueError(
            "No model endpoint configured: set api_url/api_key on SuiteConfig "
            "or API_BASE_URL/OPENCODE_API_KEY (or equivalents) in the environment or .env."
        )

    p1_dataset, p2_dataset = datasets if datasets is not None else load_probe_datasets(cfg)
    problog_res, clingo_res = controls if controls is not None else run_classical_controls(p1_dataset, p2_dataset, cfg.mode)

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
        "problog_control": problog_res,
        "clingo_control": clingo_res,
        "probe1_models": all_p1_results,
        "probe2_models": all_p2_results,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description="Run empirical reasoning probes.")
    parser.add_argument("--mode", choices=["all", "probe1", "probe2"], default="all", help="Which probe to execute")
    parser.add_argument("--dataset", choices=["standard", "seed"], default="standard", help="Benchmark dataset: 'standard' (GSM8K/ProntoQA) or 'seed' (handcrafted suite)")
    parser.add_argument("--n-items", type=int, default=20, help="Number of items to evaluate per probe")
    parser.add_argument("--api-url", default=None, help="Base URL for OpenAI-compatible endpoint")
    parser.add_argument("--api-key", default=None, help="API key for endpoint")
    parser.add_argument("--model-id", default=None, help="Model ID for API endpoint")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 = greedy)")
    parser.add_argument("--max-tokens", type=int, default=384, help="Completion token budget")
    parser.add_argument("--no-scaffold", action="store_true", help="Skip the Bayesian harness condition")
    parser.add_argument("--output-json", default=str(RESULTS_DIR / "probe_results.json"), help="Output path for raw JSON results")
    args = parser.parse_args()

    print("==========================================================================")
    print("EMPIRICAL PROBES FOR REASONING SYSTEMS (SECTION 6 EVALUATION SUITE)")
    print("==========================================================================")

    cfg = SuiteConfig(
        mode=args.mode,
        dataset=args.dataset,
        n_items=args.n_items,
        api_url=args.api_url,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        include_scaffold=not args.no_scaffold,
    )
    cfg = resolve_endpoint(cfg)
    model_id = args.model_id or os.getenv("OPENCODE_MODEL_ID") or os.getenv("MODEL_ID")
    if not model_id:
        raise ValueError(
            "No model endpoint configured: pass --model-id "
            "(or set OPENCODE_MODEL_ID/MODEL_ID in the environment or .env)."
        )

    payload = run_suite([model_id], cfg)
    if payload["errors"]:
        print(f"\n[!] Completed with {len(payload['errors'])} error(s): {', '.join(payload['errors'])}")

    with open(args.output_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[+] Raw experiment logs saved to: {args.output_json}")


if __name__ == "__main__":
    main()
