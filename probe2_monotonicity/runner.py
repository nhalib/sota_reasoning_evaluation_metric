"""Evaluation runner for Probe 2 (Monotonicity and Defeasible Stability Suite)."""

from typing import Dict, Any, List

from common.models import BaseReasoningModel
from common.schemas import Probe2Answer, ExtractionError
from .dataset import Probe2Item, PROBE2_DATASET


def format_prompt(premises: str, question: str, structured: bool = False) -> str:
    if structured:
        fmt = (
            "Respond with a JSON object with fields: "
            "verdict (boolean: true for Yes, false for No), rationale (string)."
        )
    else:
        fmt = "State your deductive reasoning, then clearly write 'Final Answer: Yes' or 'Final Answer: No'."
    return (
        f"Premises:\n{premises}\n\n"
        f"Question:\n{question}\n\n"
        f"{fmt}"
    )


def _verdict(out) -> str:
    """Normalized verdict for flip comparison: prefer schema, fall back to text.
    In fallback mode an unrecognized answer raises rather than counting as 'no'."""
    structured = getattr(out, "structured", None)
    if structured is not None:
        return "yes" if structured.verdict else "no"
    raw = out.extracted_answer.strip().lower().rstrip(".")
    if raw in ("yes", "true"):
        return "yes"
    if raw in ("no", "false"):
        return "no"
    raise ExtractionError(
        f"Unrecognized answer {raw!r} and no structured verdict available; refusing to count it as a flip."
    )


def evaluate_model_probe2(
    model: BaseReasoningModel, dataset: List[Probe2Item] = PROBE2_DATASET
) -> Dict[str, Any]:
    """Runs Probe 2 on a given target model across the dataset."""
    cond_flips = {"conjunct": 0, "instantiation": 0, "paraphrase": 0}
    item_records = []
    use_structured = getattr(model, "structured_output", False)
    schema = Probe2Answer if use_structured else None

    for item in dataset:
        # 1. Base query
        base_prompt = format_prompt(item.base_premises, item.question, structured=use_structured)
        base_out = model.generate(base_prompt, schema=schema)
        base_ans = _verdict(base_out)

        # 2. Perturbed queries for each condition
        flips = {}
        perturbed_answers = {}
        for cond_name, addition_text in item.additions.items():
            perturbed_premises = f"{item.base_premises}\n{addition_text}"
            perturbed_prompt = format_prompt(perturbed_premises, item.question, structured=use_structured)
            perturbed_out = model.generate(perturbed_prompt, schema=schema)
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
            "structured": bool(getattr(base_out, "structured", None)),
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
