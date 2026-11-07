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

`run_master.py` is the only entry point of the suite. Run the complete
evaluation across a live reasoning model endpoint
(execute from **within** the `probes/` directory):

```bash
cd probes
/home/vasily/krono/fofo/fofo_12env/bin/python run_master.py \
    --api-url "http://localhost:8000/v1" \
    --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
```

A model is required: without `--model`/`--models-file` the controller aborts, and
without a resolvable endpoint (`--api-url`/`--api-key` or the corresponding
environment/`.env` values) `run_suite()` raises before any request is sent. Raw
output lands in `results/<slug>_probe_results.json`, where `<slug>` is the
sanitized model ID.

### Options

* **Run Probe 1 only (Uncertainty):**
  ```bash
  /home/vasily/krono/fofo/fofo_12env/bin/python run_master.py --mode probe1 --model <ID> --api-url <URL>
  ```
* **Run Probe 2 only (Monotonicity):**
  ```bash
  /home/vasily/krono/fofo/fofo_12env/bin/python run_master.py --mode probe2 --model <ID> --api-url <URL>
  ```

`run_probes.py` is a library, not a script: it holds the evaluation core
(`run_suite()` plus the dataset loading helpers) and has no CLI. `python
run_probes.py` does nothing. Import it to drive the suite from your own code:

```python
from run_probes import SuiteConfig, resolve_endpoint, run_suite

cfg = resolve_endpoint(SuiteConfig(mode="probe1", dataset="seed", n_items=8))
payload = run_suite(["deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"], cfg)
```

---

## 2b. Multi-Model Sweeps (Master Controller)

`run_master.py` sweeps multiple models in-process via `run_suite()` from
`run_probes.py` — the same code path a single `--model` run uses. Datasets are
model-independent, so they are loaded **once** per unique (dataset, mode,
n_items) signature and reused across the sweep. (Classical controls ProbLog and
Clingo are verified separately via unit tests in `tests/test_probes.py`).
A failing model never aborts the run: its exception + traceback is captured in
the manifest under `error_details`, status becomes `partial`/`failed`, and the
sweep continues. Per-model outputs land in `results/<slug>_probe_results.json`;
a combined manifest (status + headline metrics) lands in
`results/sweep_results.json`.

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
`temperature`, `max_tokens`, `timeout`, `token_ceiling`, `include_scaffold` / `no_scaffold`.
Top-level defaults in `models.json`: `default_api_url`, `default_timeout`, `default_token_ceiling`.

Controller flags: `--model` (repeatable) and `--models-file` select the models;
`--mode`, `--dataset`, `--n-items`, `--api-url`, `--api-key`,
`--temperature`, `--max-tokens`, `--timeout`, `--token-ceiling`, `--no-scaffold` are sweep
defaults. Others:
* `--timeout <float>`: HTTP client read timeout in seconds (default: `180.0`; configurable via `PROBE_TIMEOUT`).
* `--token-ceiling <int>`: maximum token ceiling for truncation retry escalation (default: `16384`; configurable via `COMPLETION_TOKEN_CEILING`).
* `--seed <int>`: random seed for deterministic sampling from GSM8K and ProntoQA (default: `42`; pass `-1` for sequential slice).
* `--timestamp` / `--no-timestamp` (default: enabled): saves all outputs with UTC timestamps (`YYYYMMDD_HHMMSS`) while maintaining un-timestamped canonical copies for downstream tooling.
* `--timestamp-dir`: saves runs into dedicated timestamped subdirectories inside the output directory.
* `--output-dir`: target directory for per-model and sweep outputs (default: `results/`).
* `--output-json`: custom combined manifest path.
* `--dry-run`: print planned model list and output destinations without making API calls.
* `--fail-fast`: abort sweep immediately after the first model failure. Exit code is non-zero if any model fails.

---

## 3. Structured Elicitation (Schema-Validated Outputs)

All verbalized answers are elicited exclusively through **pydantic schemas**
(`common/schemas.py`) with server-side constrained decoding (`response_format`).
There is no free-form extraction mode.

* **Probe 1:** `Probe1Answer {verdict: bool, confidence: float in [0,1], rationale: str?}`
* **Probe 2:** `Probe2Answer {verdict: bool, rationale: str?}`

Design rules:

* **Fail loud:** any missing/out-of-range/unparseable field raises `ExtractionError`. There is no silent `0.5` fallback — a parse failure can never masquerade as a probe result (e.g. 0.5+0.5 fake coherence).
* **Dual channel (Probe 1):** the sequence-likelihood arm (natural text + token logprobs) is emitted on a separate schema-free call, because logprobs of forced-JSON tokens are format-dominated artefacts. Agreement and confidence-gap between the two channels are logged in `records[].extraction_audit_h` so structured elicitation itself is auditable.

---

## 4. Dependencies

* `pydantic` (response schemas and validation) — added to the environment alongside `problog`, `clingo`, `numpy`, `scipy`, `httpx`, `pytest`.

---

## 5. Directory Structure

```
probes/
├── README.md                          # Documentation
├── run_master.py                      # Sole CLI entry point (single models and sweeps)
├── run_probes.py                      # Evaluation core: run_suite() + setup helpers (library only)
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
│   ├── <slug>_probe_results_<ts>.json # Timestamped per-model outputs (written by run_master.py)
│   ├── <slug>_probe_results.json      # Canonical latest copy
│   ├── sweep_results_<ts>.json        # Timestamped combined sweep manifest
│   ├── sweep_results.json             # Canonical latest manifest
│   └── legacy/                        # Pre-schema artifacts, kept for provenance only
│       └── README.md                  # Why they are not reproducible / which numbers are bad
└── tests/
    └── test_probes.py                 # Pytest test suite
```

---

## 6. Running Unit Tests

To verify metrics, Clingo integration, and ProbLog execution:

```bash
/home/vasily/krono/fofo/fofo_12env/bin/pytest tests/
```
