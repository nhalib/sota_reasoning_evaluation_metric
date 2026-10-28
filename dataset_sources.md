# Standard Benchmark Datasets: Provenance & Technical Specifications

This document records the exact provenance, primary source repositories, download URLs, and operational roles of the standard benchmark datasets used for the empirical evaluation suite (**Section 6** of the paper).

---

## 1. GSM8K (Grade School Math 8K) — Probe 1: Uncertainty Coherence ($A3$)

### Bibliographic Citation
* **Title:** Training Verifiers to Solve Math Word Problems
* **Authors:** Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, John Schulman
* **Affiliation:** OpenAI
* **Publication:** arXiv preprint arXiv:2110.14168 (2021)
* **License:** MIT License

### Primary Source & Repository
* **Official Repository:** [https://github.com/openai/grade-school-math](https://github.com/openai/grade-school-math)
* **Direct Raw Download URL:**  
  `https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl`
* **Local Cached Path:** `probes/data/gsm8k_test.jsonl`
* **File Size:** ~750 KB (1,319 verified problems in test split)

### Data Format & Structure
Each record is a JSON object with:
* `"question"`: Natural-language word problem requiring 2–8 multi-step arithmetic operations.
* `"answer"`: Detailed step-by-step solution terminating in `#### <integer>` (the ground-truth numerical target).

### Operational Role in Probe 1 (\S6.2)
1. **Verbalized Confidence Elicitation:** For each problem, the model is queried on hypothesis $H$ (*"The final answer is exactly [target]"*) and complementary hypothesis $\neg H$ (*"The final answer is not [target]"*).
2. **Additivity Audit ($\Delta_{\mathrm{add}}$):** Elicited probabilities $P(H)$ and $P(\neg H)$ are evaluated to test the additivity axiom:
   $$\Delta_{\mathrm{add}} = \big| P(H) + P(\neg H) - 1.0 \big|$$
3. **ProbLog Classical Control:** Problems are compiled into relational probabilistic logic programs, providing an analytical baseline that satisfies the Kolmogorov axioms ($\Delta_{\mathrm{add}} = 0.0\%$) by construction.

---

## 2. ProntoQA — Probe 2: Monotonicity & Defeasible Stability ($A4$)

### Bibliographic Citation
* **Title:** Language Models are Greedy Reasoners: A Systematic Formal Analysis of Chain-of-Thought
* **Authors:** Abulhair Saparov, He He
* **Affiliation:** New York University (NYU)
* **Publication:** Advances in Neural Information Processing Systems (NeurIPS 2022, Oral)
* **License:** MIT License

### Primary Source & Repository
* **Official Repository:** [https://github.com/asaparov/prontoqa](https://github.com/asaparov/prontoqa)
* **Direct Raw Download URL:**  
  `https://github.com/asaparov/prontoqa/raw/main/generated_ood_data.zip`
* **Local Cached Path:** `probes/data/generated_ood_data.zip`
* **File Size:** ~5.4 MB (compressed suite of multi-hop first-order deduction problems)

### Data Format & Structure
Each record in the archive provides formal deduction instances:
* `"question"`: Base premises $\Gamma$ declaring first-order rules and facts over synthetic ontologies (e.g., *"Every rompus is a grimpus and a gorpus. Fae is a rompus."*).
* `"query"`: Target theorem $C$ to be proven (e.g., *"Prove: Fae is a grimpus."*).
* `"chain_of_thought"`: The exact formal step-by-step proof tree linking premises $\Gamma$ to conclusion $C$.

### Operational Role in Probe 2 (\S6.3)
1. **Operationalizing Definition 3.1 (Monotonicity-Preserving Additions, $\Gamma \vDash \varphi$):**  
   Because the complete formal proof tree is specified, entailed additions $\varphi$ are extracted directly from the derivation chain without ambiguity:
   * **Condition A (Conjunction):** $\varphi$ is a stated conjunct from $\Gamma$ (e.g., *"Fae is a rompus"*).
   * **Condition B (Instantiation):** $\varphi$ is an intermediate derived step or instance of an active rule (e.g., *"Fae is a grimpus and a gorpus"*).
   * **Condition C (Paraphrase):** $\varphi$ is a syntactic paraphrase of a stated rule.
2. **Defeasibility / Retraction Flip Rate:** Evaluates whether appending $\varphi$ induces an in-context conclusion flip ($C_{\mathrm{perturbed}} \neq C_{\mathrm{base}}$), testing compliance with the AGM *vacuity* postulate.
3. **Clingo Classical Control:** Compiles the ontology rules into Answer Set Programming (ASP). Evaluated under the default-negation-free fragment of Clingo, verifying a $0.0\%$ flip rate analytically by construction.

---

## 3. Offline Caching & Reproducibility Guarantees

* **Automated Ingestion:** The loader script (`probes/data_loaders.py`) automatically downloads and validates the primary sources if not present locally.
* **Deterministic Offline Execution:** Once downloaded into `probes/data/`, the evaluation suite runs 100% offline without network calls.
* **Integrity:** Datasets are immutable test splits held separate from training and prompt generation.
