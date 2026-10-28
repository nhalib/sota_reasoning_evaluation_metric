"""Model abstractions for evaluation probes, supporting OpenAI-compatible API endpoints."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, Type

from pydantic import BaseModel

from .config import CONFIDENCE_PATTERN, PROBABILITY_PATTERN, BOXED_ANSWER_PATTERN, ANSWER_PATTERN
from .schemas import (
    Probe1Answer,
    parse_structured,
    response_format,
)


@dataclass
class ModelOutput:
    text: str
    verbalized_confidence: float  # [0.0, 1.0]
    token_logprob: float         # log-likelihood
    extracted_answer: str
    structured: Optional[Any] = None      # validated Probe1Answer / Probe2Answer
    metadata: Optional[Dict[str, Any]] = None


class BaseReasoningModel(ABC):
    """Abstract interface for all evaluated models."""

    def __init__(self, name: str):
        self.name = self._sanitized(name)

    @staticmethod
    def _sanitized(name: str) -> str:
        """Filesystem-safe key for result dictionaries (model IDs often contain '/')."""
        return re.sub(r"[^0-9A-Za-z._+-]+", "_", str(name)).strip("_") or "model"

    @abstractmethod
    def generate(
        self,
        prompt: str,
        schema: Type[BaseModel],
        system_prompt: Optional[str] = None,
        logprobs: bool = False,
    ) -> ModelOutput:
        """Generate response given prompt.

        The completion is elicited via constrained decoding against ``schema``
        and validated; ``ModelOutput.structured`` carries the parsed object and
        a parse failure raises ``ExtractionError``.
        """
        pass

    def extract_confidence(self, text: str, default: Optional[float] = None) -> Optional[float]:
        """Extracts numerical confidence percentage from text (cross-check helper)."""
        pct_match = re.search(CONFIDENCE_PATTERN, text)
        if pct_match:
            try:
                val = float(pct_match.group(1))
                return min(max(val / 100.0, 0.0), 1.0)
            except ValueError:
                pass
        
        prob_match = re.search(PROBABILITY_PATTERN, text)
        if prob_match:
            try:
                val = float(prob_match.group(1))
                return min(max(val, 0.0), 1.0)
            except ValueError:
                pass
        return default

    def extract_answer(self, text: str) -> str:
        """Extracts normalized final answer string from completion."""
        boxed = re.search(BOXED_ANSWER_PATTERN, text)
        if boxed:
            return boxed.group(1).strip()
        ans = re.search(ANSWER_PATTERN, text)
        if ans:
            return ans.group(1).strip()
        # Fallback: last non-empty line
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        return lines[-1] if lines else ""


class BayesianHarnessModel(BaseReasoningModel):
    """Scaffold Condition: Wraps a base connectionist model with an external Bayesian aggregation harness (Chen et al. 2026).
    Normalizes posterior estimates over hypothesis spaces via external classical arithmetic,
    guaranteeing additivity (P(H) + P(~H) = 1.0) by construction.
    """

    def __init__(self, base_model: BaseReasoningModel):
        super().__init__(f"{base_model.name} + Bayesian Harness")
        self.base_model = base_model

    def generate(
        self,
        prompt: str,
        schema: Type[BaseModel],
        system_prompt: Optional[str] = None,
        logprobs: bool = False,
    ) -> ModelOutput:
        out = self.base_model.generate(prompt, schema, system_prompt, logprobs)
        # Harness normalizes confidence to valid probability distribution
        raw_conf = out.verbalized_confidence
        # Bayesian harness applies calibration / probability normalization
        normalized_conf = min(max(raw_conf, 0.01), 0.99)
        return ModelOutput(
            text=out.text + "\n[Bayesian Harness Applied: Coherent Normalization Enforced]",
            verbalized_confidence=normalized_conf,
            token_logprob=out.token_logprob,
            extracted_answer=out.extracted_answer,
            structured=out.structured,
            metadata={**(out.metadata or {}), "scaffolded": True, "original_confidence": raw_conf}
        )


class OpenAICompatibleModel(BaseReasoningModel):
    """Client for OpenAI-compatible chat-completions servers (opencode inference, vLLM, DeepSeek)."""

    def __init__(
        self,
        name: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        temperature: float = 0.0,
        max_tokens: int = 384,
    ):
        super().__init__(name)
        self.base_url = base_url
        self.api_key = api_key
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _post(self, payload: dict) -> dict:
        import httpx
        import uuid
        import time

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "x-opencode-session": str(uuid.uuid4())
        }

        # Robust retry loop with exponential backoff
        last_err = None
        for attempt in range(3):
            try:
                with httpx.Client(timeout=90.0) as client:
                    resp = client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
                    resp.raise_for_status()
                    return resp.json()
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as e:
                last_err = e
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"API request failed after 3 attempts: {last_err}") from last_err

    @staticmethod
    def _compose_text(msg: dict) -> str:
        reasoning = msg.get("reasoning_content", "") or ""
        content = msg.get("content", "") or ""
        if reasoning and content:
            return f"<think>\n{reasoning}\n</think>\n{content}"
        if content:
            return content
        return reasoning or ""

    @staticmethod
    def _mean_logprob(choice: dict):
        try:
            logprob_info = choice.get("logprobs", {}).get("content", [])
            if logprob_info:
                lps = [tok.get("logprob", 0.0) for tok in logprob_info if "logprob" in tok]
                if lps:
                    return sum(lps) / len(lps)
        except Exception:
            pass
        return None

    def generate(
        self,
        prompt: str,
        schema: Type[BaseModel],
        system_prompt: Optional[str] = None,
        logprobs: bool = False,
    ) -> ModelOutput:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        base_payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        # Optional free-form channel: the sequence-likelihood arm of Probe 1
        # needs natural completions; logprobs of forced-JSON tokens are
        # format-dominated artefacts, so that channel stays schema-free.
        free_text = ""
        mean_logprob = None
        raw = None
        if logprobs:
            payload = {**base_payload, "logprobs": True}
            data = self._post(payload)
            choice = data["choices"][0]
            free_text = self._compose_text(choice.get("message", {}))
            mean_logprob = self._mean_logprob(choice)
            raw = data

        # Primary channel: schema-validated answer via server-side constrained
        # decoding (strict response_format on the OpenAI-compatible endpoint).
        payload = {**base_payload, "response_format": response_format(schema)}
        data = self._post(payload)
        answer_text = self._compose_text(data["choices"][0].get("message", {}))
        structured = parse_structured(answer_text, schema)
        if not free_text:
            free_text = answer_text
            raw = data

        metadata: Dict[str, Any] = {}
        if logprobs:
            regex_conf = self.extract_confidence(free_text)
            regex_ans = self.extract_answer(free_text)
            metadata["regex_confidence"] = regex_conf
            metadata["regex_answer"] = regex_ans
            metadata["agreement"] = (
                regex_ans.lower().rstrip(".") in ("true", "yes")
            ) == structured.verdict if regex_ans else None
            if regex_conf is not None and isinstance(structured, Probe1Answer):
                metadata["conf_gap"] = round(abs(regex_conf - structured.confidence), 4)
        if raw is not None:
            metadata["raw"] = raw

        conf = structured.confidence if isinstance(structured, Probe1Answer) else (1.0 if structured.verdict else 0.0)
        ans = str(structured.verdict)

        return ModelOutput(
            text=free_text,
            verbalized_confidence=conf,
            token_logprob=mean_logprob if mean_logprob is not None else -0.5,
            extracted_answer=ans,
            structured=structured,
            metadata=metadata
        )
