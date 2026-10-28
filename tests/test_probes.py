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
    flip_rate,
    expected_calibration_error
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

    # Out-of-range confidence must raise (never a silent default)
    with pytest.raises(ExtractionError):
        parse_structured('{"verdict": true, "confidence": 1.7}', Probe1Answer)

    # Missing field must raise
    with pytest.raises(ExtractionError):
        parse_structured('{"verdict": true}', Probe1Answer)

    # No JSON at all must raise
    with pytest.raises(ExtractionError):
        parse_structured("Final Answer: True", Probe1Answer)

    # Strict response_format: all fields required, no extras
    rf = response_format(Probe1Answer)
    schema = rf["json_schema"]["schema"]
    assert rf["json_schema"]["strict"] is True
    assert set(schema["required"]) == {"verdict", "confidence", "rationale"}
    assert schema["additionalProperties"] is False

    # Probe2 verdict-only schema
    p3 = parse_structured('{"verdict": true}', Probe2Answer)
    assert p3.verdict is True
