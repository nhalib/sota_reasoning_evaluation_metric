"""Evaluation runner for Probe 2 (Monotonicity and Defeasible Stability Suite)."""

from typing import Dict, Any, List

from common.models import BaseReasoningModel
from common.schemas import Probe2Answer
from .dataset import Probe2Item, PROBE2_DATASET


def format_prompt(premises: str, question: str) -> str:
    return (
        f"Premises:\n{premises}\n\n"
        f"Question:\n{question}\n\n"
        "Respond with a JSON object with fields: "
        "verdict (boolean: true for Yes, false for No), rationale (string)."
    )


def _verdict(out) -> str:
    """Normalized verdict for flip comparison from the schema-validated answer."""
    return "yes" if out.structured.verdict else "no"


def evaluate_model_probe2(
    model: BaseReasoningModel, dataset: List[Probe2Item] = PROBE2_DATASET
) -> Dict[str, Any]:
    """Runs Probe 2 on a given target model across the dataset."""
    cond_flips = {"conjunct": 0, "instantiation": 0, "paraphrase": 0}
    item_records = []

    for item in dataset:
        # 1. Base query
        base_prompt = format_prompt(item.base_premises, item.question)
        base_out = model.generate(base_prompt, Probe2Answer)
        base_ans = _verdict(base_out)

        # 2. Perturbed queries for each condition
        flips = {}
        perturbed_answers = {}
        for cond_name, addition_text in item.additions.items():
            perturbed_premises = f"{item.base_premises}\n{addition_text}"
            perturbed_prompt = format_prompt(perturbed_premises, item.question)
            perturbed_out = model.generate(perturbed_prompt, Probe2Answer)
            perturbed_ans = _verdict(perturbed_out)
            perturbed_answers[cond_name] = perturbed_ans

            # Record flip
            is_flip = base_ans != perturbed_ans
            flips[cond_name] = is_flip
            if is_flip:
                cond_flips[cond_name] += 1

        item_records.append({
            "item_id": item.item_id,
            "base_answer": base_ans,
            "perturbed_answers": perturbed_answers,
            "flips": flips
        })

    n = len(dataset)
    rates = {cond: round((count / n) * 100.0, 1) for cond, count in cond_flips.items()}
    overall_flip_rate = round(sum(cond_flips.values()) / (n * len(cond_flips)) * 100.0, 1) if n > 0 else 0.0

    return {
        "system": model.name,
        "total_items": n,
        "flip_rates_by_condition": rates,
        "overall_flip_rate": overall_flip_rate,
        "records": item_records
    }
