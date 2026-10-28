# Empirical Probes for Reasoning Systems (Section 6)

This repository contains the standalone, open-source empirical evaluation suite operationalizing **Section 6** of the paper:
* **Probe 1 (\S6.2): Uncertainty Coherence vs. Sequence Likelihood** (evaluates $A3 = \text{none}$).
* **Probe 2 (\S6.3): Monotonicity and Defeasible Stability Suite** (evaluates $A4 = \text{unregulated}$).

Both probes benchmark connectionist reasoning models against classical formal positive controls (**ProbLog** for probabilistic coherence, and **Clingo** for deductive monotonicity).

---

## 1. Environment & Setup

The suite runs under Python 3.12 within the designated project environment:

```bash
source /home/vasily/krono/fofo/fofo_12env/bin/activate
```

Dependencies installed in the environment:
* `problog` (Probabilistic logic programming exact inference engine)
* `clingo` (Answer Set Programming grounder and solver)
* `numpy`, `scipy`, `httpx`, `pytest`

---

## 2. Quickstart

Run the complete evaluation suite across all classical controls and a live reasoning model endpoint (execute from **within** the `probes/` directory):

```bash
cd probes
/home/vasily/krono/fofo/fofo_12env/bin/python run_probes.py \
    --api-url "http://localhost:8000/v1" \
    --model-id "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
```

A model endpoint is required: if neither `--model-id`/`--api-url` nor the corresponding environment/`.env` values are provided, the run aborts with an exception. Raw outputs are captured in `results/probe_results.json`.

### Options

* **Run Probe 1 only (Uncertainty):**
  ```bash
  /home/vasily/krono/fofo/fofo_12env/bin/python run_probes.py --mode probe1 --model-id <ID> --api-url <URL>
  ```
* **Run Probe 2 only (Monotonicity):**
  ```bash
  /home/vasily/krono/fofo/fofo_12env/bin/python run_probes.py --mode probe2 --model-id <ID> --api-url <URL>
  ```

---

## 3. Directory Structure

```
probes/
├── README.md                          # Documentation
├── run_probes.py                      # Master CLI runner & Table 6 formatter
├── common/
│   ├── config.py                      # Global configuration and extraction regexes
│   ├── metrics.py                     # Brier score, additivity violation, flip rates
│   └── models.py                      # Model interfaces (Base, Bayesian Harness, API)
├── probe1_uncertainty/
│   ├── dataset.py                     # GSM8K / MATH items with paired hypotheses (H, ~H)
│   ├── problog_control.py             # Classical control: exact ProbLog solver (0% violation)
│   └── runner.py                      # Probe 1 execution logic
├── probe2_monotonicity/
│   ├── dataset.py                     # Logical premises with 3 addition conditions
│   ├── clingo_control.py              # Classical control: ASP Clingo solver (0% flips)
│   └── runner.py                      # Probe 2 execution logic
├── results/
│   └── probe_results.json             # Full raw JSON run outputs
└── tests/
    └── test_probes.py                 # Pytest test suite
```

---

## 4. Running Unit Tests

To verify metrics, Clingo integration, and ProbLog execution:

```bash
/home/vasily/krono/fofo/fofo_12env/bin/pytest tests/
```
