"""Statistical metrics for Probe 1 (Uncertainty) and Probe 2 (Monotonicity)."""

from typing import List, Tuple, Sequence
import numpy as np


def brier_score(probabilities: Sequence[float], outcomes: Sequence[int]) -> float:
    """Computes the Brier score: mean squared error between forecast probabilities and binary outcomes.
    
    BS = (1/N) * sum((p_i - y_i)^2)
    Lower is better (0.0 = perfect deterministic calibration).
    """
    probs = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if len(probs) == 0 or len(y) == 0:
        return 0.0
    return float(np.mean((probs - y) ** 2))


def additivity_violation(p_h: float, p_not_h: float) -> float:
    """Computes absolute violation of probability additivity over mutually exclusive hypotheses:
    
    Delta_add = |P(H) + P(not H) - 1.0|
    """
    return abs(float(p_h) + float(p_not_h) - 1.0)


def mean_additivity_violation_rate(pairs: Sequence[Tuple[float, float]], tolerance: float = 0.05) -> Tuple[float, float]:
    """Computes:
    1. Average absolute violation magnitude: mean(|P(H) + P(not H) - 1.0|)
    2. Rate of violation (> tolerance)
    """
    if not pairs:
        return 0.0, 0.0
    violations = [additivity_violation(p_h, p_not_h) for p_h, p_not_h in pairs]
    mean_mag = float(np.mean(violations))
    violation_rate = float(np.mean([1.0 if v > tolerance else 0.0 for v in violations]))
    return mean_mag, violation_rate


def flip_rate(base_answers: Sequence[str], perturbed_answers: Sequence[str]) -> float:
    """Computes the proportion of instances where conclusion changes under perturbation:
    
    Flip Rate = (1/N) * sum(1 if base_i != perturbed_i else 0)
    """
    if not base_answers:
        return 0.0
    flips = [
        1.0 if str(b).strip().lower() != str(p).strip().lower() else 0.0
        for b, p in zip(base_answers, perturbed_answers)
    ]
    return float(np.mean(flips))
