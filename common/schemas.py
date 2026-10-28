"""Pydantic response schemas for structured (constrained-decoding) elicitation.

These schemas are the single source of truth for how a model must report its
result. A missing/unparseable field raises ``ExtractionError`` instead of
silently defaulting, so a probe outcome can never be an artefact of failed text
parsing (the confound this module exists to remove).
"""

from typing import Optional
import json
import re

from pydantic import BaseModel, Field, ValidationError


class ExtractionError(ValueError):
    """Raised when a model response cannot be parsed into the required schema."""


class Probe1Answer(BaseModel):
    """Uncertainty probe: verdict on H plus the model's own probability for H."""

    verdict: bool = Field(description="True if the hypothesis is entailed by the premises, else False.")
    confidence: float = Field(ge=0.0, le=1.0, description="Subjective probability in [0,1] that the hypothesis is true.")
    rationale: Optional[str] = Field(default=None, description="Brief step-by-step justification.")


class Probe2Answer(BaseModel):
    """Monotonicity probe: crisp entailment verdict under base / perturbed premises."""

    verdict: bool = Field(description="True if 'Yes' follows from the premises, False for 'No'.")
    rationale: Optional[str] = Field(default=None, description="Brief step-by-step justification.")


def _strict_schema(model: type[BaseModel]) -> dict:
    """Builds an OpenAI/vLLM strict json_schema: no extras, every field required."""
    schema = model.model_json_schema()
    schema["additionalProperties"] = False
    schema["required"] = list(schema.get("properties", {}).keys())
    return schema


def response_format(model: type[BaseModel], strict: bool = True) -> dict:
    """Wraps a pydantic model into an OpenAI-compatible ``response_format`` payload."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": model.__name__,
            "strict": strict,
            "schema": _strict_schema(model),
        },
    }


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_json_object(text: str) -> dict:
    """Pulls the first JSON object from a completion, tolerating markdown fences."""
    fenced = _JSON_FENCE.search(text)
    candidate = fenced.group(1) if fenced else text
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ExtractionError(f"No JSON object found in response: {text[:200]!r}")
    try:
        obj = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError as e:
        raise ExtractionError(f"Malformed JSON in response: {e}") from e
    if not isinstance(obj, dict):
        raise ExtractionError(f"Parsed JSON is not an object: {type(obj).__name__}")
    return obj


def parse_structured(text: str, model: type[BaseModel]) -> BaseModel:
    """Validates a completion against a probe schema; raises ``ExtractionError`` on any failure."""
    obj = extract_json_object(text)
    try:
        return model.model_validate(obj)
    except ValidationError as e:
        raise ExtractionError(f"Response failed {model.__name__} validation: {e}") from e
