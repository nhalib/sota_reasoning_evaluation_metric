# Replication Suite: "Reasoning Systems, Scored Alike"
### Empirical Evaluation Suite for Section 6 and Table 4

This repository contains the standalone, open-source empirical evaluation suite supporting the paper:
> **"Reasoning Systems, Scored Alike: A Four-Axis Evaluation of Classical and Connectionist Architectures"**

It operationalizes the paper's two empirical probes auditing the unmapped taxonomic coordinates against by-construction classical controls:
* **Probe 1 (§6.2, Table 4 Column P1): Uncertainty Coherence vs. Sequence Likelihood** (evaluates $A3 = \text{none}$).
* **Probe 2 (§6.3, Table 4 Column P2): Monotonicity and Defeasible Stability Suite** (evaluates $A4 = \text{unregulated}$).
* **Classical Positive Controls (§6.1):** **ProbLog** (exact Bayesian inference) and **Clingo** (Answer Set Programming grounder and solver).

---

## 1. Paper-to-Code Traceability Matrix

| Paper Section & Artifact | Research Objective / Formal Claim | Implementation & Control Scripts | Evaluation Data & Outputs |
| :--- | :--- | :--- | :--- |
| **§6.1 (Experimental Setup)** | Standardized remote inference & dual cohort | `common/models.py`, `common/schemas.py` | Pydantic JSON schemas, deterministic decoding ($\tau=0$) |
| **§6.2 & Table 4 (Column P1)** | Kolmogorov finite additivity ($P(H) + P(\neg H) = 1.0$) | `probe1_uncertainty/runner.py`, `dataset.py` | GSM8K matched hypothesis pairs ($N=100$) |
| **§6.2 & Table 4 (Column P1)** | Classical normative probabilistic control | `probe1_uncertainty/problog_control.py` | ProbLog solver ($0.0\%$ violation rate) |
| **§6.2 & Table 4 (Column P1)** | External Bayesian scaffolding | `common/models.py` (`BayesianScaffoldedModel`) | Restores additivity; induces calibration--coherence trade-off |
| **§6.3 & Table 4 (Column P2)** | Defeasible stability under vacuous premise additions | `probe2_monotonicity/runner.py`, `dataset.py` | ProntoQA OOD ontology ($N=300$ per model) |
| **§6.3 & Table 4 (Column P2)** | Classical deductive monotonic control | `probe2_monotonicity/clingo_control.py` | Clingo ASP solver ($0.0\%$ flip rate) |
| **§6.4 & Table 4** | Complete empirical logs for the 8 evaluated models | `results/` | `results/latest_completed_evaluations.json` |

---

## 2. Quickstart: Replicating Paper Findings

### A. Verify Classical Positive Controls (Offline, No API Keys Required)
The paper's formal positive controls (**ProbLog** for Kolmogorov additivity and **Clingo** for monotonic deductive stability) can be verified locally and deterministically via the test suite:

```bash
pytest tests/
```
* **Expected Result for Probe 1 (ProbLog):** Additivity violation $\Delta_{\mathrm{add}} = 0.0000$ ($0.0\%$ violation rate) across all query pairs.
* **Expected Result for Probe 2 (Clingo):** Flip rate $= 0.0\%$ across Conjunct, Instantiation, and Semantic Paraphrase conditions.

### B. Inspect & Re-aggregate Table 4 Results from Raw Execution Logs
All raw model completion logs and structured evaluation records reported in Table 4 are archived in `results/`. To re-extract and inspect the exact numbers reported in the paper:

```bash
python collect_latest_snapshots.py
```
This generates `results/latest_completed_evaluations.json`, compiling the verified metrics for all 8 reported architectures:
1. `GLM-5.3`
2. `DeepSeek-V4.1-Flash`
3. `DeepSeek-V4-Flash`
4. `NVIDIA-Nemotron-3.5-Super-VL`
5. `Llama-3.3-70B-Instruct-Turbo`
6. `Qwen3.8-Flash`
7. `Mistral-Small-3.2-24B-Instruct-2506`
8. `Mistral-Nemo-Instruct-2407`

---

## 3. Environment & Installation

### Requirements
* Python 3.10+
* Linux / macOS

```bash
# 1. Clone the repository
git clone <repo-url>
cd probes

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install problog clingo pydantic numpy scipy httpx pytest
```

---

## 4. Running Live Evaluation Sweeps

`run_master.py` is the unified master controller for executing probes across live model endpoints.

### Executing a Single Model Run:
```bash
python run_master.py \
    --api-url "https://api.example.com/v1" \
    --api-key "$API_KEY" \
    --model "deepseek-ai/DeepSeek-V4.1-Flash"
```

### Running Targeted Probes:
* **Run Probe 1 only (Uncertainty Coherence):**
  ```bash
  python run_master.py --mode probe1 --model <MODEL_ID> --api-url <URL>
  ```
* **Run Probe 2 only (Defeasible Monotonicity):**
  ```bash
  python run_master.py --mode probe2 --model <MODEL_ID> --api-url <URL>
  ```

### Multi-Model Sweeps:
To evaluate multiple models under standardized conditions, specify models via `models.json` or CLI flags:
```bash
python run_master.py --models-file models.json
```

---

## 5. Structured Elicitation & Measurement Protocol

To eliminate extraction bias and enforce strict schema compliance without subjective parsing heuristics:
* **Pydantic Response Schemas (`common/schemas.py`):**
  * **Probe 1:** `Probe1Answer {verdict: bool, confidence: float in [0,1], rationale: str?}`
  * **Probe 2:** `Probe2Answer {verdict: bool, rationale: str?}`
* **Fail-Loud Enforcement:** Any missing, out-of-range, or unparseable field immediately raises `ExtractionError`. No silent `0.5` defaults are permitted, ensuring that parse failures cannot artificially mimic probabilistic coherence.
* **Dual-Channel Measurement:** Elicited verbalized confidence is measured alongside sequence token log-probabilities to evaluate the unassisted continuation likelihood channel independently of prompt-induced format constraints.

---

## 6. Directory Structure

```
probes/
├── README.md                          # Paper replication and execution guide
├── run_master.py                      # Master CLI controller for live probe execution
├── run_probes.py                      # Evaluation core library: run_suite() + orchestration
├── collect_latest_snapshots.py        # Snapshot aggregator generating Table 4 metrics
├── models.json                        # Declarative model sweep specifications
├── common/
│   ├── config.py                      # Global configuration parameters
│   ├── metrics.py                     # Brier score, additivity violation, flip rates
│   ├── models.py                      # Base API client and BayesianScaffoldedModel
│   └── schemas.py                     # Pydantic schemas for grammar-constrained decoding
├── probe1_uncertainty/
│   ├── dataset.py                     # GSM8K matched pairs (H, ~H) loader
│   ├── problog_control.py             # Classical control: exact ProbLog solver (0% violation)
│   └── runner.py                      # Probe 1 execution logic
├── probe2_monotonicity/
│   ├── dataset.py                     # ProntoQA synthetic premise-addition loader
│   ├── clingo_control.py              # Classical control: ASP Clingo solver (0% flips)
│   └── runner.py                      # Probe 2 execution logic
├── results/
│   ├── latest_completed_evaluations.json # Aggregated Table 4 metrics across all 8 models
│   ├── sweep_results.json             # Canonical sweep manifest
│   ├── <model_slug>_probe_results.json # Full evaluation logs for each evaluated model
│   └── legacy/                        # Historical pre-schema logs
└── tests/
    └── test_probes.py                 # Pytest test suite for classical controls & metrics
```
