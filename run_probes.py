#!/usr/bin/env python3
"""Master execution script for Empirical Probes (Section 6).
Runs Probe 1 (Uncertainty Coherence) and Probe 2 (Monotonicity Defeasibility)
against classical positive controls and a live reasoning model endpoint.
Writes the raw evaluation output to a JSON document.
"""

import sys
import os
import json
import argparse
from pathlib import Path

# Add project root (probes/) to path so the suite runs from within probes/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.config import RESULTS_DIR
from common.models import (
    BaseReasoningModel,
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


def resolve_api_config(args) -> tuple:
    """Resolves endpoint config from CLI args, falling back to environment."""
    api_url = args.api_url or os.getenv("OPENCODE_BASE_URI") or os.getenv("OPENAI_BASE_URL") or os.getenv("API_BASE_URL")
    api_key = args.api_key or os.getenv("OPENCODE_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or "EMPTY"
    model_id = args.model_id or os.getenv("OPENCODE_MODEL_ID") or os.getenv("MODEL_ID")
    if not api_url or not model_id:
        raise ValueError(
            "No model endpoint configured: pass --api-url and --model-id "
            "(or set API_BASE_URL/MODEL_ID in the environment or .env)."
        )
    return api_url, api_key, model_id


def main():
    parser = argparse.ArgumentParser(description="Run empirical reasoning probes.")
    parser.add_argument("--mode", choices=["all", "probe1", "probe2"], default="all", help="Which probe to execute")
    parser.add_argument("--dataset", choices=["standard", "seed"], default="standard", help="Benchmark dataset: 'standard' (GSM8K/ProntoQA) or 'seed' (handcrafted suite)")
    parser.add_argument("--n-items", type=int, default=20, help="Number of items to evaluate per probe")
    parser.add_argument("--api-url", default=None, help="Base URL for OpenAI-compatible endpoint")
    parser.add_argument("--api-key", default=None, help="API key for endpoint")
    parser.add_argument("--model-id", default=None, help="Model ID for API endpoint")
    parser.add_argument("--output-json", default=str(RESULTS_DIR / "probe_results.json"), help="Output path for raw JSON results")
    args = parser.parse_args()

    print("==========================================================================")
    print("EMPIRICAL PROBES FOR REASONING SYSTEMS (SECTION 6 EVALUATION SUITE)")
    print("==========================================================================")

    # 0. Load Dataset
    if args.dataset == "standard":
        print(f"\n[*] Loading Standard Benchmarks ({args.n_items} items each: GSM8K for P1, ProntoQA for P2)...")
        p1_dataset = load_standard_gsm8k(n_items=args.n_items) if args.mode in ["all", "probe1"] else []
        p2_dataset = load_standard_prontoqa(n_items=args.n_items) if args.mode in ["all", "probe2"] else []
    else:
        print("\n[*] Loading Handcrafted Seed Datasets...")
        p1_dataset = PROBE1_DATASET
        p2_dataset = PROBE2_DATASET

    # 1. Classical Controls
    problog_res = None
    clingo_res = None

    if args.mode in ["all", "probe1"]:
        print("\n[*] Executing Probe 1 Classical Control: ProbLog Exact Solver...")
        problog_res = run_problog_control(p1_dataset)
        print(f"    -> ProbLog Additivity Violation Rate: {problog_res['additivity_violation_rate']*100:.1f}% (Coherent by construction)")

    if args.mode in ["all", "probe2"]:
        print("\n[*] Executing Probe 2 Classical Control: Clingo (ASP Deductive Fragment)...")
        clingo_res = run_clingo_control(p2_dataset)
        print(f"    -> Clingo Flip Rate: {clingo_res['overall_flip_rate']:.1f}% (Monotone by construction)")

    # 2. Target Models (live endpoint required; no simulation fallback)
    api_url, api_key, model_id = resolve_api_config(args)
    print(f"\n[*] Initializing API Target Model: {model_id} via {api_url}...")
    api_model = OpenAICompatibleModel(name=model_id, base_url=api_url, api_key=api_key, model_id=model_id)
    scaffold_api = BayesianHarnessModel(api_model)
    target_models = [api_model, scaffold_api]

    all_p1_results = {}
    all_p2_results = {}

    for model in target_models:
        print(f"\n--- Evaluating Model: {model.name} ---")
        if args.mode in ["all", "probe1"]:
            p1_out = evaluate_model_probe1(model, p1_dataset)
            all_p1_results[model.name] = p1_out
            print(f"    [Probe 1] Verbalized Brier: {p1_out['verbalized_brier']} | Logprob Brier: {p1_out['logprob_brier']} | Additivity Viol: {p1_out['additivity_violation_rate']}%")

        if args.mode in ["all", "probe2"] and not getattr(model, "base_model", None):
            p2_out = evaluate_model_probe2(model, p2_dataset)
            all_p2_results[model.name] = p2_out
            rf = p2_out['flip_rates_by_condition']
            print(f"    [Probe 2] Flip Rates: Conjunct={rf['conjunct']}% | Inst={rf['instantiation']}% | Para={rf['paraphrase']}% (Overall={p2_out['overall_flip_rate']}%)")

    # 3. Save JSON results
    payload = {
        "run_config": {
            "mode": args.mode,
            "dataset": args.dataset,
            "n_items": args.n_items,
            "api_url": api_url,
            "model_id": model_id
        },
        "problog_control": problog_res,
        "clingo_control": clingo_res,
        "probe1_models": all_p1_results,
        "probe2_models": all_p2_results
    }
    with open(args.output_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[+] Raw experiment logs saved to: {args.output_json}")


if __name__ == "__main__":
    main()
