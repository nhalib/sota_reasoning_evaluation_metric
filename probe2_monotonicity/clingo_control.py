"""Classical Positive Control: Exact monotonic deduction using Answer Set Programming (Clingo).
Evaluates the default-negation-free deductive fragment, demonstrating 0.0% flip rate by construction.
"""

from typing import Dict, Any, List, Set
import clingo

from .dataset import Probe2Item, PROBE2_DATASET


def solve_asp(program: str) -> Set[str]:
    """Solves an ASP program and returns the set of stable model atoms."""
    ctl = clingo.Control(["--warn=none"])
    ctl.add("base", [], program)
    ctl.ground([("base", [])])
    atoms = set()
    with ctl.solve(yield_=True) as handle:
        for model in handle:
            for atom in model.symbols(shown=True):
                atoms.add(str(atom))
    return atoms


def evaluate_clingo_item(item: Probe2Item) -> Dict[str, Any]:
    """Evaluates an item with Clingo under base and all three addition conditions."""
    base_atoms = solve_asp(item.clingo_base)
    base_has_target = "query_target" in base_atoms

    flips = {}
    for cond_name, addition_code in item.clingo_additions.items():
        augmented_program = item.clingo_base + "\n" + addition_code
        augmented_atoms = solve_asp(augmented_program)
        augmented_has_target = "query_target" in augmented_atoms
        
        # A flip occurs if base conclusion changes under monotonicity-preserving addition
        is_flip = base_has_target != augmented_has_target
        flips[cond_name] = is_flip

    return {
        "item_id": item.item_id,
        "base_derived": base_has_target,
        "flips": flips,
        "is_monotone": not any(flips.values())
    }


def run_clingo_control(dataset: List[Probe2Item] = PROBE2_DATASET) -> Dict[str, Any]:
    """Runs the Clingo control suite across the dataset."""
    results = [evaluate_clingo_item(item) for item in dataset]
    
    cond_flips = {"conjunct": 0, "instantiation": 0, "paraphrase": 0}
    for r in results:
        for cond, flipped in r["flips"].items():
            if flipped:
                cond_flips[cond] += 1

    n = len(dataset)
    rates = {cond: (count / n) * 100.0 for cond, count in cond_flips.items()}
    overall_flip_rate = sum(cond_flips.values()) / (n * len(cond_flips)) * 100.0 if n > 0 else 0.0

    return {
        "system": "clingo (ASP Deductive Control)",
        "total_items": n,
        "flip_rates_by_condition": rates,  # All 0.0% by construction
        "overall_flip_rate": overall_flip_rate,  # 0.0%
        "item_results": results
    }
