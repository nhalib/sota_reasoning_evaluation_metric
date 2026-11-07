"""Unit tests for empirical probes suite."""

import sys
from pathlib import Path
# Add project root (probes/) to path so tests run from within probes/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np

from common.metrics import (
    brier_score,
    additivity_violation,
    mean_additivity_violation_rate,
    flip_rate
)
from common.models import BaseReasoningModel, BayesianHarnessModel, ModelOutput
from common.schemas import Probe1Answer, Probe2Answer, ExtractionError, parse_structured, response_format
from probe1_uncertainty.problog_control import run_problog_control, evaluate_problog_item
from probe1_uncertainty.dataset import PROBE1_DATASET
from probe2_monotonicity.clingo_control import run_clingo_control, evaluate_clingo_item
from probe2_monotonicity.dataset import PROBE2_DATASET


def test_metrics():
    # Brier score: perfect vs inverse
    assert brier_score([1.0, 0.0], [1, 0]) == 0.0
    assert brier_score([0.0, 1.0], [1, 0]) == 1.0
    assert round(brier_score([0.5, 0.5], [1, 0]), 2) == 0.25

    # Additivity violation
    assert additivity_violation(0.6, 0.4) == 0.0
    assert round(additivity_violation(0.9, 0.8), 2) == 0.70

    # Flip rate
    assert flip_rate(["yes", "no"], ["yes", "no"]) == 0.0
    assert flip_rate(["yes", "no"], ["no", "yes"]) == 1.0
    assert flip_rate(["yes", "yes"], ["yes", "no"]) == 0.5


def test_problog_control():
    """Verify that ProbLog exact control achieves 0.0% additivity violation."""
    res = run_problog_control(PROBE1_DATASET)
    assert res["additivity_violation_rate"] == 0.0
    assert res["mean_additivity_violation"] < 1e-6
    for item_res in res["item_results"]:
        assert item_res["is_coherent"] is True
        assert abs(item_res["p_h"] + item_res["p_not_h"] - 1.0) < 1e-6


def test_clingo_control():
    """Verify that Clingo deductive fragment achieves 0.0% flip rate by construction."""
    res = run_clingo_control(PROBE2_DATASET)
    assert res["overall_flip_rate"] == 0.0
    for cond, rate in res["flip_rates_by_condition"].items():
        assert rate == 0.0
    for item_res in res["item_results"]:
        assert item_res["is_monotone"] is True


def test_scaffold_model():
    """Verify the Bayesian harness scaffold normalizes and flags wrapped outputs."""
    class StubModel(BaseReasoningModel):
        def generate(self, prompt, schema, system_prompt=None, logprobs=False):
            text = '{"verdict": true, "confidence": 1.4}'
            return ModelOutput(
                text=text,
                verbalized_confidence=1.4,
                token_logprob=-0.4,
                extracted_answer="True",
                structured=parse_structured('{"verdict": true, "confidence": 0.9}', schema),
            )

    model = StubModel("stub")
    out = model.generate("Premises: Test.\nHypothesis: H.\nQuestion: Is it true?", Probe1Answer)
    assert isinstance(out.structured, Probe1Answer)

    # Test scaffold condition
    scaffold = BayesianHarnessModel(model)
    s_out = scaffold.generate("Premises: Test.\nHypothesis: H.\nQuestion: Is it true?", Probe1Answer)
    assert s_out.metadata["scaffolded"] is True
    assert 0.0 <= s_out.verbalized_confidence <= 1.0


def test_schema_parsing():
    """Structured elicitation must validate strictly and fail loud on bad output."""
    # Clean JSON
    p = parse_structured('{"verdict": true, "confidence": 0.82}', Probe1Answer)
    assert p.verdict is True and p.confidence == 0.82

    # Markdown-fenced JSON
    p2 = parse_structured('Here is my answer:\n```json\n{"verdict": false, "confidence": 0.3}\n```', Probe1Answer)
    assert p2.verdict is False

    # JSON with <think> scratchpad block (reasoning models)
    p_think = parse_structured(
        '<think>\nI should output {"verdict": false}\nWait, no, let me re-evaluate.\n</think>\n'
        '{"verdict": true, "confidence": 0.95, "rationale": "verified"}',
        Probe1Answer,
    )
    assert p_think.verdict is True and p_think.confidence == 0.95

    # Out-of-range confidence must raise (never a silent default)
    with pytest.raises(ExtractionError):
        parse_structured('{"verdict": true, "confidence": 1.7}', Probe1Answer)

    # Missing field must raise
    with pytest.raises(ExtractionError):
        parse_structured('{"verdict": true}', Probe1Answer)

    # No JSON at all must raise
    with pytest.raises(ExtractionError):
        parse_structured("Final Answer: True", Probe1Answer)

    # Strict response_format: only genuinely required fields (no Optional ones),
    # so a model can return a minimal object without overflowing max_tokens.
    rf = response_format(Probe1Answer)
    schema = rf["json_schema"]["schema"]
    assert rf["json_schema"]["strict"] is True
    assert set(schema["required"]) == {"verdict", "confidence"}
    assert "rationale" not in schema["required"]
    assert "rationale" in schema["properties"]
    assert schema["additionalProperties"] is False

    # Probe2 verdict-only schema
    p3 = parse_structured('{"verdict": true}', Probe2Answer)
    assert p3.verdict is True


def test_truncated_completion_retry_and_error():
    """Primary channel: escalate max_tokens on finish_reason='length'; error when bounded."""
    from common.models import OpenAICompatibleModel, TruncatedResponseError

    good = {"choices": [{"message": {"content": '{"verdict": true, "confidence": 0.8}'},
                          "finish_reason": "stop"}]}
    cut = {"choices": [{"message": {"content": '{"verdict": true, "conf'},
                        "finish_reason": "length"}]}

    # Recovers after one truncation by doubling the token cap.
    caps = []
    def post_retry(payload):
        caps.append(payload["max_tokens"])
        return cut if len(caps) == 1 else good
    m = OpenAICompatibleModel(name="m", model_id="m", max_tokens=100)
    m._post = post_retry
    out = m.generate("p", Probe1Answer)
    assert out.structured.verdict is True
    assert caps == [100, 200]

    # Persistent truncation raises a clear error rather than a misleading parse failure.
    def post_always_cut(payload):
        return cut
    m2 = OpenAICompatibleModel(name="m2", model_id="m2", max_tokens=100)
    m2._post = post_always_cut
    with pytest.raises(TruncatedResponseError):
        m2.generate("p", Probe1Answer)


def test_standard_data_loaders():
    """Verify that GSM8K and ProntoQA loaders return valid natural language items."""
    from data_loaders import load_standard_gsm8k, load_standard_prontoqa

    gsm_items = load_standard_gsm8k(n_items=3)
    assert len(gsm_items) == 3
    for it in gsm_items:
        assert it.item_id.startswith("gsm8k_")
        assert len(it.premises) > 0
        assert "exactly" in it.hypothesis_h
        assert "not" in it.hypothesis_not_h
        assert it.ground_truth_h in (0, 1)

    pronto_items = load_standard_prontoqa(n_items=3)
    assert len(pronto_items) == 3
    for it in pronto_items:
        assert it.item_id.startswith("prontoqa_")
        assert len(it.base_premises) > 0
        assert set(it.additions.keys()) == {"conjunct", "instantiation", "paraphrase"}
        assert it.expected_conclusion == "Yes"


def test_probe_runners_standalone():
    """Verify probe runners execute cleanly against a mock model without solver calls."""
    from probe1_uncertainty.runner import evaluate_model_probe1
    from probe2_monotonicity.runner import evaluate_model_probe2

    class MockModel(BaseReasoningModel):
        def generate(self, prompt, schema, system_prompt=None, logprobs=False):
            if schema == Probe1Answer:
                return ModelOutput(
                    text='{"verdict": true, "confidence": 0.8}',
                    verbalized_confidence=0.8,
                    token_logprob=-0.2,
                    extracted_answer="True",
                    structured=Probe1Answer(verdict=True, confidence=0.8),
                )
            return ModelOutput(
                text='{"verdict": true}',
                verbalized_confidence=1.0,
                token_logprob=-0.1,
                extracted_answer="True",
                structured=Probe2Answer(verdict=True),
            )

    model = MockModel("mock_evaluator")

    # Run Probe 1
    p1_res = evaluate_model_probe1(model, PROBE1_DATASET[:2])
    assert "verbalized_brier" in p1_res
    assert "additivity_violation_rate" in p1_res
    assert len(p1_res["records"]) == 2

    # Run Probe 2
    p2_res = evaluate_model_probe2(model, PROBE2_DATASET[:2])
    assert "overall_flip_rate" in p2_res
    assert "flip_rates_by_condition" in p2_res
    assert p2_res["overall_flip_rate"] == 0.0  # Constant True verdict never flips


def test_master_spec_parsing():
    """Verify load_specs in run_master correctly handles multiple models and deduplication."""
    from types import SimpleNamespace
    from run_master import load_specs, make_cfg

    args = SimpleNamespace(
        models_file=None,
        model=["test-model-1", "test-model-2", "test-model-1"],
        mode="all",
        dataset="standard",
        n_items=10,
        api_url="http://mock-api/v1",
        api_key="mock-key",
        temperature=0.0,
        max_tokens=256,
        include_scaffold=True,
    )
    specs = load_specs(args)
    assert len(specs) == 2  # Duplicate test-model-1 ignored
    assert [s["model_id"] for s in specs] == ["test-model-1", "test-model-2"]

    cfg = make_cfg(specs[0], args)
    assert cfg.n_items == 10
    assert cfg.api_url == "http://mock-api/v1"


def test_models_json_default_resolution():
    """Verify that models.json is automatically discovered and resolves default_api_url."""
    from types import SimpleNamespace
    from run_master import load_specs, make_cfg

    # Empty CLI args
    args = SimpleNamespace(
        models_file=None,
        model=[],
        mode="all",
        dataset="standard",
        n_items=20,
        api_url=None,
        api_key=None,
        temperature=0.0,
        max_tokens=384,
        include_scaffold=True,
    )
    specs = load_specs(args)
    assert len(specs) >= 1
    # Check that the first model inherited the default_api_url from models.json
    first_model = specs[0]
    assert first_model["api_url"] == "https://api.deepinfra.com/v1/openai"

    cfg = make_cfg(first_model, args)
    assert cfg.api_url == "https://api.deepinfra.com/v1/openai"


def test_timestamped_outputs(tmp_path):
    """Verify timestamp formatting and output path resolution."""
    import re
    from common.config import get_timestamp_str
    from run_master import resolve_model_output_paths, resolve_sweep_output_paths

    ts = get_timestamp_str()
    assert re.match(r"^\d{8}_\d{6}$", ts)

    # 1. Model output paths with timestamping (default)
    p_primary, p_latest = resolve_model_output_paths("test_model", tmp_path, timestamp=ts)
    assert p_primary == tmp_path / f"test_model_probe_results_{ts}.json"
    assert p_latest == tmp_path / "test_model_probe_results.json"

    # 2. Model output paths with timestamping disabled
    p_notime, p_notime_latest = resolve_model_output_paths("test_model", tmp_path, timestamp=None)
    assert p_notime == tmp_path / "test_model_probe_results.json"
    assert p_notime_latest is None

    # 3. Model output paths inside dedicated timestamp directory
    p_dir_primary, p_dir_latest = resolve_model_output_paths("test_model", tmp_path, timestamp=ts, timestamp_dir=True)
    assert p_dir_primary == tmp_path / ts / f"test_model_probe_results_{ts}.json"
    assert p_dir_latest == tmp_path / ts / "test_model_probe_results.json"
    assert (tmp_path / ts).is_dir()

    # 4. Sweep manifest paths default
    s_primary, s_latest = resolve_sweep_output_paths(None, tmp_path, timestamp=ts)
    assert s_primary == tmp_path / f"sweep_results_{ts}.json"
    assert s_latest == tmp_path / "sweep_results.json"

    # 5. Sweep manifest paths with custom file and timestamp
    custom_target = str(tmp_path / "custom_manifest.json")
    s_cust_primary, s_cust_latest = resolve_sweep_output_paths(custom_target, tmp_path, timestamp=ts)
    assert s_cust_primary == tmp_path / f"custom_manifest_{ts}.json"
    assert s_cust_latest == Path(custom_target)


def test_seeded_dataset_sampling():
    """Verify deterministic seeded sampling across GSM8K and ProntoQA."""
    from data_loaders import load_standard_gsm8k, load_standard_prontoqa

    # GSM8K
    gsm_seed42_a = load_standard_gsm8k(n_items=5, seed=42)
    gsm_seed42_b = load_standard_gsm8k(n_items=5, seed=42)
    gsm_seed99 = load_standard_gsm8k(n_items=5, seed=99)
    gsm_seq = load_standard_gsm8k(n_items=5, seed=None)

    assert len(gsm_seed42_a) == 5
    assert [x.premises for x in gsm_seed42_a] == [x.premises for x in gsm_seed42_b]
    assert [x.premises for x in gsm_seed42_a] != [x.premises for x in gsm_seed99]
    assert [x.premises for x in gsm_seed42_a] != [x.premises for x in gsm_seq]

    # ProntoQA
    pqa_seed42_a = load_standard_prontoqa(n_items=5, seed=42)
    pqa_seed42_b = load_standard_prontoqa(n_items=5, seed=42)
    pqa_seed99 = load_standard_prontoqa(n_items=5, seed=99)
    pqa_seq = load_standard_prontoqa(n_items=5, seed=None)

    assert len(pqa_seed42_a) == 5
    assert [x.base_premises for x in pqa_seed42_a] == [x.base_premises for x in pqa_seed42_b]
    assert [x.base_premises for x in pqa_seed42_a] != [x.base_premises for x in pqa_seed99]
    assert [x.base_premises for x in pqa_seed42_a] != [x.base_premises for x in pqa_seq]


def test_timeout_and_token_ceiling_config():
    """Verify timeout and token ceiling defaults, propagation, and retry bounds."""
    from common.config import DEFAULT_TIMEOUT, DEFAULT_TOKEN_CEILING
    from common.models import OpenAICompatibleModel, TruncatedResponseError
    from run_probes import SuiteConfig, build_models
    from run_master import make_cfg
    from types import SimpleNamespace

    # 1. Defaults
    assert DEFAULT_TIMEOUT == 180.0
    assert DEFAULT_TOKEN_CEILING == 16384
    m_default = OpenAICompatibleModel(name="test", model_id="test")
    assert m_default.timeout == 180.0
    assert m_default.token_ceiling == 16384

    # 2. Custom values on OpenAICompatibleModel
    m_custom = OpenAICompatibleModel(
        name="test", model_id="test", timeout=240.0, token_ceiling=32768, truncation_retries=5
    )
    assert m_custom.timeout == 240.0
    assert m_custom.token_ceiling == 32768
    assert m_custom.truncation_retries == 5

    # 3. Propagation through SuiteConfig and build_models
    cfg = SuiteConfig(api_url="http://localhost:8000/v1", api_key="test", timeout=200.0, token_ceiling=8192)
    models = build_models(cfg, "test_model")
    assert models[0].timeout == 200.0
    assert models[0].token_ceiling == 8192

    # 4. make_cfg propagation from specs and defaults
    defaults = SimpleNamespace(
        mode="all", dataset="standard", n_items=10, seed=42,
        api_url="http://localhost:8000/v1", api_key="key",
        temperature=0.0, max_tokens=500,
        timeout=150.0, token_ceiling=10000,
        include_scaffold=False,
    )
    spec_no_override = {"model_id": "m1"}
    cfg1 = make_cfg(spec_no_override, defaults)
    assert cfg1.timeout == 150.0
    assert cfg1.token_ceiling == 10000

    spec_with_override = {"model_id": "m2", "timeout": 300.0, "token_ceiling": 20000}
    cfg2 = make_cfg(spec_with_override, defaults)
    assert cfg2.timeout == 300.0
    assert cfg2.token_ceiling == 20000

    # 5. Token escalation bounds up to custom token_ceiling
    cut = {"choices": [{"message": {"content": '{"verdict": true, "conf'}, "finish_reason": "length"}]}
    caps = []
    def record_cap(payload):
        caps.append(payload["max_tokens"])
        return cut

    m_escalate = OpenAICompatibleModel(
        name="esc", model_id="esc", max_tokens=1000, token_ceiling=3500, truncation_retries=4
    )
    m_escalate._post = record_cap
    with pytest.raises(TruncatedResponseError):
        m_escalate.generate("prompt", Probe1Answer)
    # Expected progression: 1000 -> 2000 -> 3500 (capped by token_ceiling) -> TruncatedResponseError
    assert caps == [1000, 2000, 3500]




