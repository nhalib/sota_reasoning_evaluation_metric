"""Classical Positive Control: Exact probabilistic logic execution using ProbLog.
Demonstrates analytical probability coherence and 0% additivity violation.
"""

from typing import Dict, Any, List
import problog
from problog.program import PrologString
from problog import get_evaluatable
from problog.logic import Term

from .dataset import Probe1Item, PROBE1_DATASET


def evaluate_problog_item(item: Probe1Item) -> Dict[str, Any]:
    """Evaluates an item using the ProbLog exact solver engine."""
    model = PrologString(item.problog_program)
    results = get_evaluatable().create_from(model).evaluate()
    
    # Extract queried probabilities
    p_h = 0.0
    p_not_h = 0.0

    # In ProbLog, results dictionary maps Term -> float probability
    for k, v in results.items():
        k_str = str(k).strip()
        if k_str == "h":
            p_h = float(v)
        elif k_str == "not_h":
            p_not_h = float(v)

    # Calculate additivity violation
    delta_add = abs(p_h + p_not_h - 1.0)

    return {
        "item_id": item.item_id,
        "p_h": p_h,
        "p_not_h": p_not_h,
        "delta_add": delta_add,
        "is_coherent": delta_add < 1e-6
    }


def run_problog_control(dataset: List[Probe1Item] = PROBE1_DATASET) -> Dict[str, Any]:
    """Runs the ProbLog control over the entire dataset."""
    results = [evaluate_problog_item(item) for item in dataset]
    violations = [r["delta_add"] for r in results]
    mean_violation = sum(violations) / len(violations) if violations else 0.0
    violation_rate = sum(1.0 for v in violations if v > 1e-4) / len(violations) if violations else 0.0

    return {
        "system": "ProbLog (Classical Control)",
        "total_items": len(dataset),
        "mean_additivity_violation": mean_violation,
        "additivity_violation_rate": violation_rate,  # Exactly 0.0% by construction
        "item_results": results
    }
