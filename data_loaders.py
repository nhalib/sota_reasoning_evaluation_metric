"""Standard benchmark dataset loaders for empirical probes.

Ingests:
1. GSM8K (Cobbe et al., 2021) for Probe 1 (Uncertainty Coherence)
2. ProntoQA (Saparov & He, NeurIPS 2022) for Probe 2 (Monotonicity & Defeasible Stability)
Caches raw files locally in probes/data/ for reproducible, offline evaluation.
"""

import os
import re
import json
import random
import zipfile
from pathlib import Path
from typing import List, Optional

import httpx

from common.config import PROBES_DIR, DEFAULT_SEED
from probe1_uncertainty.dataset import Probe1Item
from probe2_monotonicity.dataset import Probe2Item

DATA_DIR = PROBES_DIR / "data"
os.makedirs(DATA_DIR, exist_ok=True)

GSM8K_TEST_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
PRONTOQA_ZIP_URL = "https://github.com/asaparov/prontoqa/raw/main/generated_ood_data.zip"


def load_standard_gsm8k(n_items: int = 50, seed: Optional[int] = DEFAULT_SEED) -> List[Probe1Item]:
    """Downloads and formats standard verifiable math items from GSM8K test split for Probe 1.

    If seed is not None, deterministically samples n_items from the test split.
    If seed is None, takes the first n_items sequentially.
    """
    cache_file = DATA_DIR / "gsm8k_test.jsonl"
    if not cache_file.exists():
        print(f"[*] Downloading GSM8K test set from {GSM8K_TEST_URL}...")
        resp = httpx.get(GSM8K_TEST_URL, timeout=30.0)
        resp.raise_for_status()
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write(resp.text)

    with open(cache_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    valid_entries = []
    for line in lines:
        entry = json.loads(line)
        raw_answer = entry.get("answer", "").strip()
        num_match = re.search(r"####\s*(-?[0-9\.,]+)", raw_answer)
        if num_match:
            ans_num = num_match.group(1).replace(",", "").strip()
            valid_entries.append((entry["question"].strip(), ans_num))

    if seed is not None:
        rng = random.Random(seed)
        selected = rng.sample(valid_entries, min(n_items, len(valid_entries)))
    else:
        selected = valid_entries[:n_items]

    items: List[Probe1Item] = []
    for idx, (question, ans_num) in enumerate(selected):
        item_id = f"gsm8k_{idx:03d}"
        hypothesis_h = f"The final numerical answer to this problem is exactly {ans_num}."
        hypothesis_not_h = f"The final numerical answer to this problem is not {ans_num}."

        items.append(Probe1Item(
            item_id=item_id,
            premises=question,
            hypothesis_h=hypothesis_h,
            hypothesis_not_h=hypothesis_not_h,
            ground_truth_h=1,
        ))

    print(f"[+] Loaded {len(items)} standard GSM8K items for Probe 1 (seed={seed}).")
    return items


def load_standard_prontoqa(n_items: int = 50, seed: Optional[int] = DEFAULT_SEED) -> List[Probe2Item]:
    """Downloads and formats first-order deductive reasoning items from ProntoQA for Probe 2.

    If seed is not None, deterministically samples n_items from the candidate pool.
    If seed is None, takes the first n_items sequentially.
    """
    cache_file = DATA_DIR / "generated_ood_data.zip"
    if not cache_file.exists():
        print(f"[*] Downloading ProntoQA dataset from {PRONTOQA_ZIP_URL}...")
        resp = httpx.get(PRONTOQA_ZIP_URL, timeout=30.0, follow_redirects=True)
        resp.raise_for_status()
        with open(cache_file, "wb") as f:
            f.write(resp.content)

    candidate_items: List[dict] = []
    with zipfile.ZipFile(cache_file) as z:
        target_files = sorted([f for f in z.namelist() if "3testhops" in f or "2testhops" in f or "random" in f])
        if not target_files:
            target_files = sorted(z.namelist())

        for filename in target_files:
            data = json.loads(z.read(filename).decode("utf-8"))
            for ex_key in sorted(data.keys()):
                ex_data = data[ex_key]
                test_ex = ex_data.get("test_example")
                if not test_ex:
                    continue

                question_text = test_ex["question"].strip()
                query_text = test_ex["query"].replace("Prove:", "").strip()
                proof_steps = test_ex.get("chain_of_thought", [])

                if not proof_steps:
                    continue

                conjunct_addition = f"Additionally, we know that {proof_steps[0].rstrip('.')}."
                inst_step = proof_steps[1] if len(proof_steps) > 1 else proof_steps[0]
                inst_addition = f"Additionally, note that {inst_step.rstrip('.')}."
                para_addition = f"Additionally, it is given as an established rule that {proof_steps[0].rstrip('.')}."

                candidate_items.append({
                    "base_premises": question_text,
                    "question": f"Does it follow that {query_text}? Answer with 'Yes' or 'No'.",
                    "expected_conclusion": "Yes",
                    "additions": {
                        "conjunct": conjunct_addition,
                        "instantiation": inst_addition,
                        "paraphrase": para_addition
                    }
                })

    if seed is not None:
        rng = random.Random(seed)
        selected = rng.sample(candidate_items, min(n_items, len(candidate_items)))
    else:
        selected = candidate_items[:n_items]

    items: List[Probe2Item] = [
        Probe2Item(
            item_id=f"prontoqa_{idx:03d}",
            base_premises=cand["base_premises"],
            question=cand["question"],
            expected_conclusion=cand["expected_conclusion"],
            additions=cand["additions"],
        )
        for idx, cand in enumerate(selected)
    ]

    print(f"[+] Loaded {len(items)} standard ProntoQA items for Probe 2 (seed={seed}).")
    return items
