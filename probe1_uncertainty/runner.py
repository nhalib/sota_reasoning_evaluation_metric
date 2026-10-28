"""Evaluation runner for Probe 1 (Uncertainty Coherence vs. Sequence Likelihood)."""

import math
from typing import Dict, Any, List
import numpy as np

from common.models import BaseReasoningModel
from common.metrics import brier_score, additivity_violation
from common.schemas import Probe1Answer
from .dataset import Probe1Item, PROBE1_DATASET


def format_prompt(premises: str, hypothesis: str) -> str:
    return (
        f"Premises:\n{premises}\n\n"
        f"Hypothesis:\n{hypothesis}\n\n"
        "Question: Is this hypothesis true? Evaluate the premises and provide your conclusion.\n"
        "Respond with a JSON object with fields: "
        "verdict (boolean), confidence (number in [0,1] = your subjective probability that the hypothesis is true), "
        "rationale (string)."
    )


def evaluate_model_probe1(
    model: BaseReasoningModel, dataset: List[Probe1Item] = PROBE1_DATASET
) -> Dict[str, Any]:
    """Runs Probe 1 on a given target model across the dataset."""
    verbalized_probs = []
    logprob_probs = []
    ground_truths = []
    pairs = []
    item_records = []

    for item in dataset:
        # 1. Query for Hypothesis H (schema-validated; logprob channel stays unconstrained)
        prompt_h = format_prompt(item.premises, item.hypothesis_h)
        out_h = model.generate(prompt_h, Probe1Answer, logprobs=True)
        conf_h = out_h.verbalized_confidence
        # Map logprob to probability space [0, 1] via logistic sigmoid: sigma(logprob + 1.0)
        prob_from_logprob_h = 1.0 / (1.0 + math.exp(-min(max(out_h.token_logprob + 1.0, -5.0), 5.0)))

        # 2. Query for Hypothesis Not H (mutually exclusive complement)
        prompt_not_h = format_prompt(item.premises, item.hypothesis_not_h)
        out_not_h = model.generate(prompt_not_h, Probe1Answer, logprobs=True)
        conf_not_h = out_not_h.verbalized_confidence

        # If evaluating a Bayesian harness scaffold, the harness normalizes over the binary partition
        if getattr(model, "is_scaffold", False) or hasattr(model, "base_model"):
            total_mass = conf_h + conf_not_h
            if total_mass > 0:
                conf_h = conf_h / total_mass
                conf_not_h = 1.0 - conf_h

        # Record metrics
        verbalized_probs.append(conf_h)
        logprob_probs.append(prob_from_logprob_h)
        ground_truths.append(item.ground_truth_h)

        pairs.append((conf_h, conf_not_h))
        delta_add = additivity_violation(conf_h, conf_not_h)

        item_records.append({
            "item_id": item.item_id,
            "conf_h": conf_h,
            "conf_not_h": conf_not_h,
            "delta_add": delta_add,
            "verdict_h": out_h.structured.verdict,
            "logprob_h": out_h.token_logprob,
            "prob_from_logprob_h": prob_from_logprob_h,
            "extraction_audit_h": {
                k: (out_h.metadata or {}).get(k)
                for k in ("regex_confidence", "conf_gap")
            },
            "ground_truth": item.ground_truth_h
        })

    # Summary statistics
    brier_verb = brier_score(verbalized_probs, ground_truths)
    brier_logp = brier_score(logprob_probs, ground_truths)
    violations = [p["delta_add"] for p in item_records]
    mean_delta = float(np.mean(violations))
    additivity_rate = float(np.mean([1.0 if v > 0.05 else 0.0 for v in violations]))

    return {
        "system": model.name,
        "verbalized_brier": round(brier_verb, 4),
        "logprob_brier": round(brier_logp, 4),
        "mean_additivity_violation": round(mean_delta, 4),
        "additivity_violation_rate": round(additivity_rate * 100.0, 1),
        "records": item_records
    }
