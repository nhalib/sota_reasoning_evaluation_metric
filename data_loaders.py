"""Standard benchmark dataset loaders for empirical probes.
Ingests:
1. GSM8K (Cobbe et al., 2021) for Probe 1 (Uncertainty Coherence)
2. ProntoQA (Saparov & He, NeurIPS 2022) for Probe 2 (Monotonicity & Defeasible Stability)
Caches raw files locally in probes/data/ for reproducible, offline evaluation.
"""

import os
import re
import json
import io
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Optional

import httpx

from common.config import PROBES_DIR
from probe1_uncertainty.dataset import Probe1Item
from probe2_monotonicity.dataset import Probe2Item

DATA_DIR = PROBES_DIR / "data"
os.makedirs(DATA_DIR, exist_ok=True)

GSM8K_TEST_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
PRONTOQA_ZIP_URL = "https://github.com/asaparov/prontoqa/raw/main/generated_ood_data.zip"


def clean_asp_symbol(s: str) -> str:
    """Converts a natural language name/concept to a valid ASP predicate/constant name."""
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s if s else "entity"


def load_standard_gsm8k(n_items: int = 50) -> List[Probe1Item]:
    """Downloads and formats standard verifiable math items from GSM8K test split for Probe 1."""
    cache_file = DATA_DIR / "gsm8k_test.jsonl"
    if not cache_file.exists():
        print(f"[*] Downloading GSM8K test set from {GSM8K_TEST_URL}...")
        resp = httpx.get(GSM8K_TEST_URL, timeout=30.0)
        resp.raise_for_status()
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write(resp.text)
    
    with open(cache_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    items: List[Probe1Item] = []
    for idx, line in enumerate(lines[:n_items]):
        entry = json.loads(line)
        question = entry["question"].strip()
        raw_answer = entry["answer"].strip()
        
        # Extract numerical answer after ####
        num_match = re.search(r"####\s*(-?[0-9\.,]+)", raw_answer)
        if not num_match:
            continue
        ans_num = num_match.group(1).replace(",", "").strip()
        
        item_id = f"gsm8k_{idx:03d}"
        hypothesis_h = f"The final numerical answer to this problem is exactly {ans_num}."
        hypothesis_not_h = f"The final numerical answer to this problem is not {ans_num}."

        # Compile minimal exact ProbLog representation
        problog_code = f"""
        1.0::target_answer({ans_num}).
        h :- target_answer({ans_num}).
        not_h :- \\+ h.
        query(h).
        query(not_h).
        """

        items.append(Probe1Item(
            item_id=item_id,
            premises=question,
            hypothesis_h=hypothesis_h,
            hypothesis_not_h=hypothesis_not_h,
            ground_truth_h=1,
            problog_program=problog_code
        ))

    print(f"[+] Loaded {len(items)} standard GSM8K items for Probe 1.")
    return items


def load_standard_prontoqa(n_items: int = 50) -> List[Probe2Item]:
    """Downloads and formats first-order deductive reasoning items from ProntoQA for Probe 2."""
    cache_file = DATA_DIR / "generated_ood_data.zip"
    if not cache_file.exists():
        print(f"[*] Downloading ProntoQA dataset from {PRONTOQA_ZIP_URL}...")
        resp = httpx.get(PRONTOQA_ZIP_URL, timeout=30.0, follow_redirects=True)
        resp.raise_for_status()
        with open(cache_file, "wb") as f:
            f.write(resp.content)

    items: List[Probe2Item] = []
    with zipfile.ZipFile(cache_file) as z:
        # Load 3-hop random files for rich deductive chains
        target_files = [f for f in z.namelist() if "3testhops" in f or "2testhops" in f or "random" in f]
        if not target_files:
            target_files = z.namelist()

        for filename in target_files:
            data = json.loads(z.read(filename).decode("utf-8"))
            for ex_key, ex_data in data.items():
                if len(items) >= n_items:
                    break
                test_ex = ex_data.get("test_example")
                if not test_ex:
                    continue

                question_text = test_ex["question"].strip()
                query_text = test_ex["query"].replace("Prove:", "").strip()
                proof_steps = test_ex.get("chain_of_thought", [])

                if not proof_steps:
                    continue

                # The first step in proof is an explicit premise/conjunct
                conjunct_addition = f"Additionally, we know that {proof_steps[0].rstrip('.')}."
                # Intermediate step in proof is an entailed instance
                inst_step = proof_steps[1] if len(proof_steps) > 1 else proof_steps[0]
                inst_addition = f"Additionally, note that {inst_step.rstrip('.')}."
                # Paraphrase addition
                para_addition = f"Additionally, it is given as an established rule that {proof_steps[0].rstrip('.')}."

                # Build ASP Clingo rules for the deductive ontology
                asp_rules = []
                for sentence in question_text.split("."):
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    # Parse "Each X is a Y and a Z" or "X is a Y"
                    m_each = re.match(r"(?:Each|Every|All)\s+([a-zA-Z0-9]+)\s+(?:is a|are)\s+([a-zA-Z0-9]+)", sentence, re.IGNORECASE)
                    if m_each:
                        sub, obj = clean_asp_symbol(m_each.group(1)), clean_asp_symbol(m_each.group(2))
                        asp_rules.append(f"{obj}(X) :- {sub}(X).")
                        continue
                    m_fact = re.match(r"([a-zA-Z0-9]+)\s+is a\s+([a-zA-Z0-9]+)", sentence, re.IGNORECASE)
                    if m_fact:
                        subj, cls = clean_asp_symbol(m_fact.group(1)), clean_asp_symbol(m_fact.group(2))
                        asp_rules.append(f"{cls}({subj}).")

                # Parse query "Fae is a grimpus"
                m_q = re.match(r"([a-zA-Z0-9]+)\s+is a\s+([a-zA-Z0-9]+)", query_text, re.IGNORECASE)
                if m_q:
                    q_subj, q_cls = clean_asp_symbol(m_q.group(1)), clean_asp_symbol(m_q.group(2))
                    asp_rules.append(f"query_target :- {q_cls}({q_subj}).")
                else:
                    asp_rules.append("query_target :- true.")
                asp_rules.append("#show query_target/0.")

                clingo_base_code = "\n".join(asp_rules)

                items.append(Probe2Item(
                    item_id=f"prontoqa_{len(items):03d}",
                    base_premises=question_text,
                    question=f"Does it follow that {query_text}? Answer with 'Yes' or 'No'.",
                    expected_conclusion="Yes",
                    additions={
                        "conjunct": conjunct_addition,
                        "instantiation": inst_addition,
                        "paraphrase": para_addition
                    },
                    clingo_base=clingo_base_code,
                    clingo_additions={
                        "conjunct": f"% conjunct addition: {proof_steps[0]}",
                        "instantiation": f"% instance addition: {inst_step}",
                        "paraphrase": f"% paraphrase rule: {proof_steps[0]}"
                    }
                ))

            if len(items) >= n_items:
                break

    print(f"[+] Loaded {len(items)} standard ProntoQA items for Probe 2.")
    return items
