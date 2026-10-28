#!/usr/bin/env python3
"""Master controller for the Empirical Probes suite.

Loads GSM8K/ProntoQA and runs the ProbLog/Clingo classical controls exactly
once, then sweeps run_probes.run_suite across multiple model IDs in-process.
A failing model is captured (traceback in the manifest) without aborting the
sweep. Per-model JSON outputs are written to results/<slug>_probe_results.json;
a combined manifest with status + headline metrics goes to
results/sweep_results.json.
"""

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

# Add project root (probes/) to path so the controller runs from within probes/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.config import RESULTS_DIR
from common.models import BaseReasoningModel
from run_probes import (
    SuiteConfig,
    resolve_endpoint,
    load_probe_datasets,
    run_classical_controls,
    run_suite,
)


def slugify(model_id: str) -> str:
    return BaseReasoningModel._sanitized(model_id)


def load_specs(args) -> list:
    """Collects model specs from --models-file and repeated --model flags.

    A spec is either a bare model ID string or an object whose keys override
    the sweep defaults (mode, dataset, n_items, api_url, api_key,
    temperature, max_tokens, include_scaffold).
    """
    specs = []
    if args.models_file:
        raw = json.loads(Path(args.models_file).read_text(encoding="utf-8"))
        entries = raw.get("models", []) if isinstance(raw, dict) else raw
        if not isinstance(entries, list):
            raise ValueError("--models-file: expected a list or {'models': [...]} object.")
        for entry in entries:
            if isinstance(entry, str):
                entry = {"model_id": entry}
            if not isinstance(entry, dict) or not entry.get("model_id"):
                raise ValueError(f"--models-file: entry missing 'model_id': {entry!r}")
            specs.append(entry)

    for model_id in args.model:
        specs.append({"model_id": model_id})

    if not specs:
        raise ValueError("No models specified: pass --models-file <json> and/or --model <id> (repeatable).")

    seen, deduped = set(), []
    for spec in specs:
        mid = spec["model_id"]
        if mid in seen:
            print(f"[!] Duplicate model_id '{mid}' ignored.")
            continue
        seen.add(mid)
        deduped.append(spec)
    return deduped


def make_cfg(spec: dict, defaults) -> SuiteConfig:
    """Builds a SuiteConfig by layering per-spec overrides on CLI defaults.

    API keys resolve: spec > --api-key > environment (.env via resolve_endpoint).
    """
    include_scaffold = defaults.include_scaffold
    if "no_scaffold" in spec:
        include_scaffold = not spec["no_scaffold"]
    if "include_scaffold" in spec:
        include_scaffold = bool(spec["include_scaffold"])

    cfg = SuiteConfig(
        mode=spec.get("mode", defaults.mode),
        dataset=spec.get("dataset", defaults.dataset),
        n_items=spec.get("n_items", defaults.n_items),
        include_scaffold=include_scaffold,
        api_url=spec.get("api_url", defaults.api_url),
        api_key=spec.get("api_key", defaults.api_key),
        temperature=spec.get("temperature", defaults.temperature),
        max_tokens=spec.get("max_tokens", defaults.max_tokens),
    )
    return resolve_endpoint(cfg)


def summarize(payload: dict) -> dict:
    """Extracts headline metrics from a run_suite payload."""
    p1_head = lambda o: {k: o[k] for k in ("verbalized_brier", "logprob_brier", "additivity_violation_rate") if k in o}
    p2_head = lambda o: {k: o[k] for k in ("flip_rates_by_condition", "overall_flip_rate") if k in o}
    return {
        "probe1_models": {n: p1_head(o) for n, o in payload.get("probe1_models", {}).items()},
        "probe2_models": {n: p2_head(o) for n, o in payload.get("probe2_models", {}).items()},
    }


def digest(summary: dict) -> str:
    parts = []
    for name, m in summary.get("probe1_models", {}).items():
        if m.get("verbalized_brier") is not None:
            parts.append(f"P1 verbBrier[{name}]={m['verbalized_brier']} additViol={m.get('additivity_violation_rate')}%")
    for name, m in summary.get("probe2_models", {}).items():
        if m.get("overall_flip_rate") is not None:
            parts.append(f"P2 flip[{name}]={m['overall_flip_rate']}%")
    return " | ".join(parts) if parts else "no metrics parsed"


def model_config_view(cfg: SuiteConfig) -> dict:
    """Config snapshot for the manifest. api_key deliberately excluded."""
    return {
        "mode": cfg.mode,
        "dataset": cfg.dataset,
        "n_items": cfg.n_items,
        "api_url": cfg.api_url,
        "include_scaffold": cfg.include_scaffold,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description="Master controller: probe sweep across multiple models.")
    parser.add_argument("--models-file", default=None,
                        help="JSON list (or {'models': [...]}) of model IDs or {model_id, ...} specs")
    parser.add_argument("--model", action="append", default=[],
                        help="Model ID (repeatable; combined with --models-file)")
    parser.add_argument("--mode", choices=["all", "probe1", "probe2"], default="all")
    parser.add_argument("--dataset", choices=["standard", "seed"], default="standard")
    parser.add_argument("--n-items", type=int, default=20)
    parser.add_argument("--api-url", default=None, help="Shared endpoint URL (per-model override via JSON)")
    parser.add_argument("--api-key", default=None, help="Shared endpoint key (never written to any file)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--no-scaffold", action="store_true", help="Skip the Bayesian harness condition")
    parser.add_argument("--fail-fast", action="store_true", help="Stop the sweep after the first failed model")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned model list without executing")
    parser.add_argument("--output-json", default=str(RESULTS_DIR / "sweep_results.json"),
                        help="Combined sweep manifest path")
    args = parser.parse_args()

    specs = load_specs(args)

    d = SimpleNamespace(
        mode=args.mode, dataset=args.dataset, n_items=args.n_items,
        api_url=args.api_url, api_key=args.api_key,
        temperature=args.temperature, max_tokens=args.max_tokens,
        include_scaffold=not args.no_scaffold,
    )

    print("==========================================================================")
    print("MASTER CONTROLLER: MULTI-MODEL PROBE SWEEP")
    print(f"Models ({len(specs)}): " + ", ".join(s["model_id"] for s in specs))
    print("==========================================================================")

    if args.dry_run:
        for spec in specs:
            cfg = make_cfg(spec, d)
            print(f"[dry-run] {spec['model_id']} -> {slugify(spec['model_id'])}_probe_results.json | {model_config_view(cfg)}")
        return

    cfgs = [make_cfg(s, d) for s in specs]

    # Datasets and classical controls are model-independent and deterministic:
    # compute them once, keyed by their config signature, and reuse.
    datasets_cache, controls_cache = {}, {}

    def get_setup(cfg: SuiteConfig):
        dkey = (cfg.dataset, cfg.mode, cfg.n_items)
        if dkey not in datasets_cache:
            print("\n[*] Loading datasets (shared across compatible models)...")
            datasets_cache[dkey] = load_probe_datasets(cfg)
        ckey = (dkey, cfg.mode)
        if ckey not in controls_cache:
            print("\n[*] Running model-independent classical controls (shared)...")
            controls_cache[ckey] = run_classical_controls(*datasets_cache[dkey], cfg.mode)
        return datasets_cache[dkey], controls_cache[ckey]

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sweep_config": {
            "model_ids": [s["model_id"] for s in specs],
            "defaults": {
                "mode": d.mode,
                "dataset": d.dataset,
                "n_items": d.n_items,
                "api_url": d.api_url,
                "include_scaffold": d.include_scaffold,
                "temperature": d.temperature,
                "max_tokens": d.max_tokens,
            },
        },
        "models": [],
    }

    failures = []
    for idx, spec in enumerate(specs, 1):
        model_id = spec["model_id"]
        cfg = cfgs[idx - 1]
        slug = slugify(model_id)
        out_path = RESULTS_DIR / f"{slug}_probe_results.json"

        print(f"\n[{idx}/{len(specs)}] {model_id}")
        t0 = time.monotonic()
        entry = {
            "model_id": model_id,
            "slug": slug,
            "config": model_config_view(cfg),
            "output_file": str(out_path),
        }

        try:
            datasets, controls = get_setup(cfg)
            tracebacks = {}
            payload = run_suite([model_id], cfg, controls=controls, datasets=datasets,
                                 on_model_error=lambda name, exc: tracebacks.__setitem__(
                                     name, "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))))
            entry["exit_code"] = 0
            if payload["errors"]:
                entry["status"] = "partial"
                entry["error_details"] = {n: {"message": msg, "traceback": tracebacks.get(n, "")}
                                          for n, msg in payload["errors"].items()}
                failures.append(model_id)
                print(f"    [PARTIAL] {len(payload['errors'])} model condition(s) errored")
            else:
                entry["status"] = "ok"
            entry["summary"] = summarize(payload)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"    [{entry['status']}] {digest(entry['summary'])}")
        except Exception as exc:
            entry["exit_code"] = 1
            entry["status"] = "failed"
            entry["error_details"] = {
                model_id: {"message": f"{type(exc).__name__}: {exc}",
                           "traceback": traceback.format_exc()}
            }
            failures.append(model_id)
            print(f"    [FAILED] {type(exc).__name__}: {exc}")

        entry["duration_s"] = round(time.monotonic() - t0, 1)
        manifest["models"].append(entry)

        if failures and args.fail_fast:
            print("\n[!] --fail-fast: aborting sweep after first failure.")
            break

    manifest["failures"] = failures
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    ok = sum(1 for e in manifest["models"] if e["status"] == "ok")
    print(f"\n[+] Sweep manifest saved to: {out_path}")
    print(f"[+] {ok}/{len(specs)} model(s) completed successfully"
          + (f"; failures: {', '.join(failures)}" if failures else "."))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
