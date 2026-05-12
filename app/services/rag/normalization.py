"""Score normalization strategies for hybrid retrieval."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple


NORMALIZATION_MIN_MAX = "min-max"
NORMALIZATION_SOFTMAX = "softmax"
NORMALIZATION_Z_SCORE = "z-score"


def _safe_float_list(values: Sequence[float]) -> List[float]:
    safe: List[float] = []
    for value in values:
        if math.isnan(value) or math.isinf(value):
            safe.append(0.0)
        else:
            safe.append(float(value))
    return safe


def normalize_scores(values: Sequence[float], strategy: str) -> Tuple[List[float], Dict[str, float]]:
    safe = _safe_float_list(values)
    if not safe:
        return [], {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}

    minimum = min(safe)
    maximum = max(safe)
    mean = sum(safe) / len(safe)
    variance = sum((x - mean) ** 2 for x in safe) / max(1, len(safe))
    std = math.sqrt(variance)

    if strategy == NORMALIZATION_MIN_MAX:
        denom = maximum - minimum
        if denom <= 1e-9:
            normalized = [0.5 for _ in safe]
        else:
            normalized = [(x - minimum) / denom for x in safe]
    elif strategy == NORMALIZATION_SOFTMAX:
        # Stable softmax with max-subtraction.
        shift = max(safe)
        exp_values = [math.exp(max(-50.0, min(50.0, x - shift))) for x in safe]
        total = sum(exp_values)
        if total <= 1e-9:
            normalized = [1.0 / len(safe) for _ in safe]
        else:
            normalized = [x / total for x in exp_values]
            # Re-scale softmax probabilities to [0, 1] for easier hybrid blending.
            min_prob = min(normalized)
            max_prob = max(normalized)
            prob_denom = max_prob - min_prob
            if prob_denom > 1e-9:
                normalized = [(x - min_prob) / prob_denom for x in normalized]
    elif strategy == NORMALIZATION_Z_SCORE:
        if std <= 1e-9:
            normalized = [0.5 for _ in safe]
        else:
            z_values = [(x - mean) / std for x in safe]
            # Map z-score to [0, 1] with sigmoid for stable blending.
            normalized = [1.0 / (1.0 + math.exp(-z)) for z in z_values]
    else:
        normalized = safe

    clipped = [max(0.0, min(1.0, value)) for value in normalized]
    return clipped, {
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "std": std,
    }
