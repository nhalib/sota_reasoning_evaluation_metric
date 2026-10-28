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

## 2b. Multi-Model Sweeps (Master Controller)

`run_master.py` sweeps multiple models in-process via `run_suite()` from
`run_probes.py`. Datasets and the ProbLog/Clingo classical controls are
model-independent, so they are computed **once** per unique (dataset, mode,
n_items) signature and reused across the sweep. A failing model never aborts
the run: its exception + traceback is captured in the manifest under
`error_details`, status becomes `partial`/`failed`, and the sweep continues.
Per-model outputs land in `results/<slug>_probe_results.json`; a combined
manifest (status + headline metrics) lands in `results/sweep_results.json`.

```bash
# Two ad-hoc models against one endpoint:
/home/vasily/krono/fofo/fofo_12env/bin/python run_master.py \
    --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" \
    --model "openai/o3-mini" \
    --api-url "https://opencode.ai/inference/openai/v1"

# Declarative spec file with per-model overrides; bare strings are also allowed:
/home/vasily/krono/fofo/fofo_12env/bin/python run_master.py --models-file models.json
```

`models.json` example (each object key overrides the CLI default for that
model only):

```json
{
  "models": [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    {"model_id": "qwen/qwq-32b", "n_items": 50},
    {"model_id": "openai/o3-mini", "mode": "probe1", "temperature": 0.7, "no_scaffold": true}
  ]
}
```

Overridable per-model keys: `mode`, `dataset`, `n_items`, `api_url`, `api_key`,
`temperature`, `max_tokens`, `structured` / `no_structured`,
`include_scaffold` / `no_scaffold`.

Controller flags: `--mode`, `--dataset`, `--n-items`, `--api-url`, `--api-key`,
`--temperature`, `--max-tokens`, `--no-structured`, `--no-scaffold` are sweep
defaults. Others: `--dry-run` (print resolved per-model configs; the API key is
never written to any output file), `--fail-fast` (stop after first failure).
Exit code is non-zero if any model fails.

Single-model runs still work via `run_probes.py` directly (same flags minus the
sweep-only ones); it calls the same `run_suite([model_id], cfg)` core.

---

## 3. Structured Elicitation (Schema-Validated Outputs)

Verbalized answers are extracted through **pydantic schemas** (`common/schemas.py`) with server-side constrained decoding (`response_format`):

* **Probe 1:** `Probe1Answer {verdict: bool, confidence: float in [0,1], rationale: str?}`
* **Probe 2:** `Probe2Answer {verdict: bool, rationale: str?}`

Design rules:

* **Fail loud:** any missing/out-of-range/unparseable field raises `ExtractionError`. There is no silent `0.5` fallback — a parse failure can never masquerade as a probe result (e.g. 0.5+0.5 fake coherence).
* **Dual channel:** the free-form arm (natural text + token logprobs, used by the sequence-likelihood Brier) is emitted on a separate schema-free call; the JSON arm elicits verbalized confidence. Per-item agreement and confidence-gap between channels are logged in `records[].extraction_audit_h` so structured elicitation itself is auditable.
* **Escape hatch:** `--no-structured` reverts to legacy regex-only extraction (with fail-loud semantics).

---

## 4. Dependencies

* `pydantic` (response schemas and validation) — added to the environment alongside `problog`, `clingo`, `numpy`, `scipy`, `httpx`, `pytest`.

---

## 5. Directory Structure

```
probes/
├── README.md                          # Documentation
├── run_master.py                      # Master controller: multi-model sweeps
├── run_probes.py                      # Evaluation core (run_suite) + single-model CLI
├── common/
│   ├── config.py                      # Global configuration and extraction regexes
│   ├── metrics.py                     # Brier score, additivity violation, flip rates
│   ├── models.py                      # Model interfaces (Base, Bayesian Harness, API)
│   └── schemas.py                     # Pydantic response schemas + strict JSON parsing
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

## 6. Running Unit Tests

To verify metrics, Clingo integration, and ProbLog execution:

```bash
/home/vasily/krono/fofo/fofo_12env/bin/pytest tests/
```
