"""Evaluation runner for Probe 2 (Monotonicity and Defeasible Stability Suite)."""

from typing import Dict, Any, List
import numpy as np

from common.models import BaseReasoningModel
from .dataset import Probe2Item, PROBE2_DATASET


def format_prompt(premises: str, question: str) -> str:
    return (
        f"Premises:\n{premises}\n\n"
        f"Question:\n{question}\n\n"
        "State your deductive reasoning, then clearly write 'Final Answer: Yes' or 'Final Answer: No'."
    )


def evaluate_model_probe2(
    model: BaseReasoningModel, dataset: List[Probe2Item] = PROBE2_DATASET
) -> Dict[str, Any]:
    """Runs Probe 2 on a given target model across the dataset."""
    cond_flips = {"conjunct": 0, "instantiation": 0, "paraphrase": 0}
    item_records = []

    for item in dataset:
        # 1. Base query
        base_prompt = format_prompt(item.base_premises, item.question)
        base_out = model.generate(base_prompt)
        base_ans = base_out.extracted_answer.strip().lower()

        # 2. Perturbed queries for each condition
        flips = {}
        for cond_name, addition_text in item.additions.items():
            perturbed_premises = f"{item.base_premises}\n{addition_text}"
            perturbed_prompt = format_prompt(perturbed_premises, item.question)
            perturbed_out = model.generate(perturbed_prompt)
            perturbed_ans = perturbed_out.extracted_answer.strip().lower()

            # Record flip
            is_flip = base_ans != perturbed_ans
            flips[cond_name] = is_flip
            if is_flip:
                cond_flips[cond_name] += 1

        item_records.append({
            "item_id": item.item_id,
            "base_answer": base_ans,
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
