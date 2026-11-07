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
from typing import Optional, Tuple

# Add project root (probes/) to path so the controller runs from within probes/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.config import (
    RESULTS_DIR,
    DEFAULT_SEED,
    DEFAULT_TIMEOUT,
    DEFAULT_TOKEN_CEILING,
    get_timestamp_str,
)
from common.models import BaseReasoningModel
from run_probes import (
    SuiteConfig,
    resolve_endpoint,
    load_probe_datasets,
    run_suite,
)


def slugify(model_id: str) -> str:
    return BaseReasoningModel._sanitized(model_id)


def resolve_model_output_paths(
    slug: str,
    output_dir: Path,
    timestamp: Optional[str] = None,
    timestamp_dir: bool = False,
) -> tuple[Path, Optional[Path]]:
    """Resolves output file path for a model.

    Returns (primary_path, latest_path):
    - primary_path: path to write the run output (timestamped if timestamp is provided).
    - latest_path: canonical un-timestamped file path (updated alongside primary if timestamped).
    """
    target_dir = output_dir / timestamp if (timestamp and timestamp_dir) else output_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    if timestamp:
        primary_path = target_dir / f"{slug}_probe_results_{timestamp}.json"
        latest_path = target_dir / f"{slug}_probe_results.json"
        return primary_path, latest_path
    else:
        primary_path = target_dir / f"{slug}_probe_results.json"
        return primary_path, None


def resolve_sweep_output_paths(
    output_json: Optional[str],
    output_dir: Path,
    timestamp: Optional[str] = None,
    timestamp_dir: bool = False,
) -> tuple[Path, Optional[Path]]:
    """Resolves output file path for the sweep manifest.

    Returns (primary_path, latest_path).
    """
    target_dir = output_dir / timestamp if (timestamp and timestamp_dir) else output_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    if output_json:
        custom_path = Path(output_json)
        custom_path.parent.mkdir(parents=True, exist_ok=True)
        if timestamp:
            stem = custom_path.stem
            suffix = custom_path.suffix or ".json"
            ts_path = custom_path.parent / f"{stem}_{timestamp}{suffix}"
            return ts_path, custom_path
        return custom_path, None

    if timestamp:
        primary_path = target_dir / f"sweep_results_{timestamp}.json"
        latest_path = target_dir / "sweep_results.json"
        return primary_path, latest_path
    else:
        primary_path = target_dir / "sweep_results.json"
        return primary_path, None


def load_specs(args) -> list:
    """Collects model specs from --models-file and repeated --model flags.

    A spec is either a bare model ID string or an object whose keys override
    the sweep defaults (mode, dataset, n_items, api_url, api_key,
    temperature, max_tokens, include_scaffold). If neither --models-file nor
    --model is passed, defaults to 'models.json' if present.
    """
    specs = []
    file_defaults = {}
    models_file = getattr(args, "models_file", None)
    if not models_file and not getattr(args, "model", None) and Path("models.json").exists():
        models_file = "models.json"

    if models_file:
        raw = json.loads(Path(models_file).read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            entries = raw.get("models", [])
            file_defaults = {k: v for k, v in raw.items() if k != "models"}
        else:
            entries = raw
        if not isinstance(entries, list):
            raise ValueError(f"{models_file}: expected a list or {{'models': [...]}} object.")
        for entry in entries:
            if isinstance(entry, str):
                entry = {"model_id": entry}
            if not isinstance(entry, dict) or not entry.get("model_id"):
                raise ValueError(f"{models_file}: entry missing 'model_id': {entry!r}")
            # Inherit file-level defaults from models.json if not overridden by CLI or spec
            if "api_url" not in entry and "default_api_url" in file_defaults and not getattr(args, "api_url", None):
                entry["api_url"] = file_defaults["default_api_url"]
            if "timeout" not in entry and "default_timeout" in file_defaults and getattr(args, "timeout", None) is None:
                entry["timeout"] = file_defaults["default_timeout"]
            if "token_ceiling" not in entry and "default_token_ceiling" in file_defaults and getattr(args, "token_ceiling", None) is None:
                entry["token_ceiling"] = file_defaults["default_token_ceiling"]
            specs.append(entry)

    for model_id in getattr(args, "model", []):
        specs.append({"model_id": model_id})

    if not specs:
        raise ValueError("No models specified: configure models.json, or pass --models-file <json> / --model <id>.")

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
        seed=spec.get("seed", getattr(defaults, "seed", DEFAULT_SEED)),
        include_scaffold=include_scaffold,
        api_url=spec.get("api_url", defaults.api_url),
        api_key=spec.get("api_key", defaults.api_key),
        temperature=spec.get("temperature", defaults.temperature),
        max_tokens=spec.get("max_tokens", defaults.max_tokens),
        timeout=spec.get("timeout", getattr(defaults, "timeout", DEFAULT_TIMEOUT)),
        token_ceiling=spec.get("token_ceiling", getattr(defaults, "token_ceiling", DEFAULT_TOKEN_CEILING)),
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
        "seed": cfg.seed,
        "api_url": cfg.api_url,
        "include_scaffold": cfg.include_scaffold,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "timeout": cfg.timeout,
        "token_ceiling": cfg.token_ceiling,
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
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed for sampling benchmark items (default: {DEFAULT_SEED}; pass -1 for sequential)")
    parser.add_argument("--api-url", default=None, help="Shared endpoint URL (per-model override via JSON)")
    parser.add_argument("--api-key", default=None, help="Shared endpoint key (never written to any file)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--timeout", type=float, default=None,
                        help=f"HTTP client read timeout in seconds (default: {DEFAULT_TIMEOUT})")
    parser.add_argument("--token-ceiling", type=int, default=None,
                        help=f"Max tokens ceiling for truncation retry escalation (default: {DEFAULT_TOKEN_CEILING})")
    parser.add_argument("--no-scaffold", action="store_true", help="Skip the Bayesian harness condition")
    parser.add_argument("--fail-fast", action="store_true", help="Stop the sweep after the first failed model")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned model list without executing")
    parser.add_argument("--output-json", default=None,
                        help="Combined sweep manifest path (default: <output-dir>/sweep_results[_<timestamp>].json)")
    parser.add_argument("--output-dir", default=str(RESULTS_DIR),
                        help=f"Base output directory for results (default: {RESULTS_DIR})")
    parser.add_argument("--timestamp", action=argparse.BooleanOptionalAction, default=True,
                        help="Save output files with timestamped filenames (default: enabled)")
    parser.add_argument("--timestamp-dir", action="store_true",
                        help="Save outputs into a dedicated timestamped subdirectory inside output-dir")
    args = parser.parse_args()

    specs = load_specs(args)

    effective_seed = None if args.seed == -1 else args.seed
    effective_timeout = args.timeout if args.timeout is not None else DEFAULT_TIMEOUT
    effective_token_ceiling = args.token_ceiling if args.token_ceiling is not None else DEFAULT_TOKEN_CEILING

    d = SimpleNamespace(
        mode=args.mode, dataset=args.dataset, n_items=args.n_items,
        seed=effective_seed,
        api_url=args.api_url, api_key=args.api_key,
        temperature=args.temperature, max_tokens=args.max_tokens,
        timeout=effective_timeout, token_ceiling=effective_token_ceiling,
        include_scaffold=not args.no_scaffold,
    )

    output_dir = Path(args.output_dir)
    run_timestamp = get_timestamp_str() if args.timestamp else None
    manifest_path, latest_manifest_path = resolve_sweep_output_paths(
        args.output_json, output_dir, run_timestamp, args.timestamp_dir
    )

    print("==========================================================================")
    print("MASTER CONTROLLER: MULTI-MODEL PROBE SWEEP")
    print(f"Models ({len(specs)}): " + ", ".join(s["model_id"] for s in specs))
    if run_timestamp:
        print(f"Run Timestamp: {run_timestamp}")
    print(f"Output Directory: {output_dir}")
    print(f"Sampling Seed: {effective_seed}")
    print("==========================================================================")

    if args.dry_run:
        for spec in specs:
            cfg = make_cfg(spec, d)
            out_p, _ = resolve_model_output_paths(
                slugify(spec["model_id"]), output_dir, run_timestamp, args.timestamp_dir
            )
            print(f"[dry-run] {spec['model_id']} -> {out_p.name} | {model_config_view(cfg)}")
        print(f"[dry-run] Sweep manifest -> {manifest_path.name}")
        return

    cfgs = [make_cfg(s, d) for s in specs]

    # Datasets are model-independent and deterministic:
    # compute them once, keyed by their config signature, and reuse.
    datasets_cache = {}

    def get_dataset(cfg: SuiteConfig):
        dkey = (cfg.dataset, cfg.mode, cfg.n_items, cfg.seed)
        if dkey not in datasets_cache:
            print(f"\n[*] Loading datasets (shared across compatible models, seed={cfg.seed})...")
            datasets_cache[dkey] = load_probe_datasets(cfg)
        return datasets_cache[dkey]

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_timestamp": run_timestamp,
        "sweep_config": {
            "model_ids": [s["model_id"] for s in specs],
            "defaults": {
                "mode": d.mode,
                "dataset": d.dataset,
                "n_items": d.n_items,
                "seed": d.seed,
                "api_url": d.api_url,
                "include_scaffold": d.include_scaffold,
                "temperature": d.temperature,
                "max_tokens": d.max_tokens,
                "timeout": d.timeout,
                "token_ceiling": d.token_ceiling,
                "timestamp": args.timestamp,
                "timestamp_dir": args.timestamp_dir,
            },
        },
        "models": [],
    }

    failures = []
    for idx, spec in enumerate(specs, 1):
        model_id = spec["model_id"]
        cfg = cfgs[idx - 1]
        slug = slugify(model_id)
        out_path, latest_path = resolve_model_output_paths(
            slug, output_dir, run_timestamp, args.timestamp_dir
        )

        print(f"\n[{idx}/{len(specs)}] {model_id}")
        t0 = time.monotonic()
        entry = {
            "model_id": model_id,
            "slug": slug,
            "config": model_config_view(cfg),
            "output_file": str(out_path),
            "latest_file": str(latest_path) if latest_path else None,
        }

        try:
            datasets = get_dataset(cfg)
            tracebacks = {}
            payload = run_suite([model_id], cfg, datasets=datasets,
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
            if latest_path and latest_path != out_path:
                with open(latest_path, "w", encoding="utf-8") as f:
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
    manifest["sweep_manifest_file"] = str(manifest_path)
    if latest_manifest_path and latest_manifest_path != manifest_path:
        manifest["latest_manifest_file"] = str(latest_manifest_path)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    if latest_manifest_path and latest_manifest_path != manifest_path:
        with open(latest_manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    ok = sum(1 for e in manifest["models"] if e["status"] == "ok")
    print(f"\n[+] Sweep manifest saved to: {manifest_path}")
    if latest_manifest_path and latest_manifest_path != manifest_path:
        print(f"[+] Latest sweep manifest updated at: {latest_manifest_path}")
    print(f"[+] {ok}/{len(specs)} model(s) completed successfully"
          + (f"; failures: {', '.join(failures)}" if failures else "."))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
