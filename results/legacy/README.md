# Legacy results — not reproducible by the current suite

These two files predate the structured-elicitation change (`46bb00f`, "all LLM
responses are structured via pydantics"). They are kept for provenance only.
**Do not cite the numbers in them.**

Nothing in the codebase reads or writes these paths; `run_master.py` writes
`results/<slug>_probe_results.json` and `results/sweep_results.json`.

| File | Contents |
|---|---|
| `probe_results.json` | Pre-schema output for four named systems: `DeepSeek-R1-Distill-14B`, `DeepSeek-R1-Distill-32B`, `QwQ-32B`, `OpenAI o1 / o3-mini` (plus one Bayesian-harness condition). Those baseline entries were removed from the suite in `19963b0`, so no current invocation can produce this document. |
| `glm53_results.json` | Pre-schema single-model run under a non-standard filename (predates the `<slug>_probe_results.json` convention). |

## How you can tell they are stale

* No `run_config` or `errors` top-level keys (added by `run_suite()` afterwards).
* Probe 1 records lack `verdict_h` and `extraction_audit_h` — i.e. they were
  produced before schema-validated elicitation and the dual-channel audit.
* Probe 2 records lack `perturbed_answers`.
* `probe_results.json` stores Probe 2 verdicts as the placeholder strings
  `correct_verified_answer` / `erroneous_answer`, not the `yes` / `no` that
  `runner._verdict()` emits today.

## Known-bad measurements

* **`glm53_results.json`, likelihood channel is a placeholder.** Every `logprob_h`
  is exactly `-0.5`. The old `OpenAICompatibleModel.generate` substituted
  `token_logprob = -0.5` when the endpoint returned no logprobs; that fallback has
  since been replaced by `None`. So `logprob_brier: 0.2038` and every
  `prob_from_logprob_h: 0.6224…` are derived from a constant, not from measured
  token logprobs.
* **`glm53_results.json`, regex-extracted verdicts.** Probe 2 `base_answer` values
  include `yes**` — free-form markdown leaking through the pre-schema extractor.

The classical-control blocks in both files are *not* suspect in the same way: they
were run on the handcrafted seed sets, and every `item_id` still matches today's
`PROBE1_DATASET` / `PROBE2_DATASET` exactly. It is the model records, and in
`glm53_results.json` the whole likelihood channel, that cannot be trusted.
