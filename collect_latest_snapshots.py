#!/usr/bin/env python3
"""
collect_latest_snapshots.py
===========================
Discovers, validates, and extracts the latest timestamped evaluation snapshots
for all models that have completed full evaluation runs across both Probe 1
(Uncertainty Coherence) and Probe 2 (Monotonicity & Defeasible Stability).

Usage:
    python3 collect_latest_snapshots.py
    python3 collect_latest_snapshots.py --output results/latest_completed_evaluations.json
"""

import argparse
import glob
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple


SNAPSHOT_REGEX = re.compile(r"^(.*)_probe_results_(\d{8}_\d{6})\.json$")


def is_full_evaluation_run(data: Dict[str, Any]) -> bool:
    """
    Validates whether a snapshot represents a full evaluation run:
    - probe1_models must contain non-empty evaluation records
    - probe2_models must contain non-empty evaluation records
    """
    p1 = data.get("probe1_models", {})
    p2 = data.get("probe2_models", {})

    if not isinstance(p1, dict) or not isinstance(p2, dict):
        return False

    if not p1 or not p2:
        return False

    # Check that at least one model in p1 and p2 has evaluated metrics/records
    has_p1_data = any(
        isinstance(v, dict) and ("verbalized_brier" in v or "records" in v)
        for v in p1.values()
    )
    has_p2_data = any(
        isinstance(v, dict) and ("overall_flip_rate" in v or "records" in v)
        for v in p2.values()
    )

    return has_p1_data and has_p2_data


def get_latest_timestamped_snapshots(results_dir: str = "results") -> Dict[str, Dict[str, Any]]:
    """
    Finds the latest timestamped snapshot for each model with a full evaluation run.

    Returns:
        Dict mapping model_id -> {
            'model_id': str,
            'timestamp': str,
            'file_path': str,
            'data': dict
        }
    """
    candidates: Dict[str, List[Tuple[str, str, Dict[str, Any]]]] = {}

    search_pattern = os.path.join(results_dir, "*_probe_results_*.json")
    for file_path in glob.glob(search_pattern):
        file_name = os.path.basename(file_path)
        match = SNAPSHOT_REGEX.match(file_name)
        if not match:
            continue

        prefix, timestamp = match.groups()

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to parse {file_path}: {e}")
            continue

        if not is_full_evaluation_run(data):
            continue

        # Extract canonical model_id from run_config or file prefix
        model_id = None
        run_config = data.get("run_config", {})
        cfg_models = run_config.get("model_ids", [])
        if cfg_models:
            model_id = cfg_models[0]
        else:
            # Fallback to key from probe2_models or file prefix
            p2_keys = list(data.get("probe2_models", {}).keys())
            model_id = p2_keys[0] if p2_keys else prefix

        if model_id not in candidates:
            candidates[model_id] = []
        candidates[model_id].append((timestamp, file_path, data))

    latest_snapshots: Dict[str, Dict[str, Any]] = {}
    for model_id, runs in candidates.items():
        # Sort descending by timestamp (YYYYMMDD_HHMMSS formats naturally sort lexicographically)
        runs.sort(key=lambda item: item[0], reverse=True)
        latest_ts, latest_path, latest_data = runs[0]
        latest_snapshots[model_id] = {
            "model_id": model_id,
            "timestamp": latest_ts,
            "file_path": latest_path,
            "all_full_snapshot_count": len(runs),
            "data": latest_data,
        }

    return latest_snapshots


def extract_model_summary(snapshot_entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts summary metrics from a single snapshot entry for quick inspection and reporting.
    """
    model_id = snapshot_entry["model_id"]
    file_path = snapshot_entry["file_path"]
    timestamp = snapshot_entry["timestamp"]
    data = snapshot_entry["data"]

    p1 = data.get("probe1_models", {})
    p2 = data.get("probe2_models", {})

    # Locate unassisted model and Bayesian harness entry in Probe 1
    unassisted_key = None
    harness_key = None

    for k in p1.keys():
        if "harness" in k.lower():
            harness_key = k
        else:
            unassisted_key = k

    p1_unassisted = p1.get(unassisted_key, {}) if unassisted_key else {}
    p1_harness = p1.get(harness_key, {}) if harness_key else {}

    # Probe 2 unassisted model
    p2_unassisted_key = list(p2.keys())[0] if p2 else None
    p2_unassisted = p2.get(p2_unassisted_key, {}) if p2_unassisted_key else {}

    return {
        "model_id": model_id,
        "timestamp": timestamp,
        "file_path": file_path,
        "probe1_unassisted": {
            "system": unassisted_key,
            "verbalized_brier": p1_unassisted.get("verbalized_brier"),
            "logprob_brier": p1_unassisted.get("logprob_brier"),
            "mean_additivity_violation": p1_unassisted.get("mean_additivity_violation"),
            "additivity_violation_rate_pct": p1_unassisted.get("additivity_violation_rate"),
        },
        "probe1_harness": {
            "system": harness_key,
            "verbalized_brier": p1_harness.get("verbalized_brier"),
            "logprob_brier": p1_harness.get("logprob_brier"),
            "mean_additivity_violation": p1_harness.get("mean_additivity_violation"),
            "additivity_violation_rate_pct": p1_harness.get("additivity_violation_rate"),
        },
        "probe2": {
            "system": p2_unassisted_key,
            "total_items": p2_unassisted.get("total_items"),
            "overall_flip_rate_pct": p2_unassisted.get("overall_flip_rate"),
            "flip_rates_by_condition": p2_unassisted.get("flip_rates_by_condition", {}),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract the latest timestamped snapshot for fully evaluated models."
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Path to results directory (default: 'results')",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Optional JSON output path to save consolidated results",
    )
    args = parser.parse_args()

    latest = get_latest_timestamped_snapshots(args.results_dir)

    if not latest:
        print("[INFO] No fully evaluated models found in timestamped snapshots.")
        return

    print("=" * 105)
    print(f"LATEST TIMESTAMPED SNAPSHOTS FOR FULLY EVALUATED MODELS ({len(latest)} MODELS FOUND)")
    print("=" * 105)
    header = (
        f"{'Model ID':<44} | {'Timestamp':<15} | "
        f"{'P1 Addit Viol':<14} | {'Harness Viol':<13} | {'P2 Flip Rate':<12}"
    )
    print(header)
    print("-" * 105)

    consolidated = {}
    for model_id, entry in sorted(latest.items()):
        summary = extract_model_summary(entry)
        consolidated[model_id] = summary

        p1_un = summary["probe1_unassisted"]
        p1_h = summary["probe1_harness"]
        p2 = summary["probe2"]

        p1_viol_str = (
            f"{p1_un['mean_additivity_violation']:.4f} ({p1_un['additivity_violation_rate_pct']:.0f}%)"
            if p1_un["mean_additivity_violation"] is not None
            else "N/A"
        )
        h_viol_str = (
            f"{p1_h['mean_additivity_violation']:.4f} ({p1_h['additivity_violation_rate_pct']:.0f}%)"
            if p1_h["mean_additivity_violation"] is not None
            else "N/A"
        )
        p2_flip_str = (
            f"{p2['overall_flip_rate_pct']:.2f}%"
            if p2["overall_flip_rate_pct"] is not None
            else "N/A"
        )

        print(
            f"{model_id:<44} | {entry['timestamp']:<15} | "
            f"{p1_viol_str:<14} | {h_viol_str:<13} | {p2_flip_str:<12}"
        )

    print("=" * 105)
    print("\nExact File Paths Selected:")
    for model_id, entry in sorted(latest.items()):
        print(f"  • {model_id}:")
        print(f"      File: {entry['file_path']}")
        print(f"      Snapshot Timestamp: {entry['timestamp']}")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(consolidated, f, indent=2)
        print(f"\n[OK] Successfully saved consolidated summary to: {args.output}")


if __name__ == "__main__":
    main()
