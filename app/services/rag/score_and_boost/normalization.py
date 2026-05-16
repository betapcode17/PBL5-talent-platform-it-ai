"""Normalization helpers used when combining heterogeneous score signals."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple


NORMALIZATION_MIN_MAX = "min_max"
NORMALIZATION_SOFTMAX = "softmax"
NORMALIZATION_Z_SCORE = "z_score"


def _stats_from_vals(vals: List[float]) -> Dict[str, float]:
	if not vals:
		return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
	mn = min(vals)
	mx = max(vals)
	mean = sum(vals) / len(vals)
	var = sum((v - mean) ** 2 for v in vals) / len(vals)
	std = math.sqrt(var) if var > 0 else 0.0
	return {"min": float(mn), "max": float(mx), "mean": float(mean), "std": float(std)}


def normalize_scores(values: Iterable[float], mode: str = NORMALIZATION_MIN_MAX) -> Tuple[List[float], Dict[str, float]]:
	"""Normalize a sequence of numeric scores and return (normalized_values, stats).

	Returns a tuple so callers can inspect distribution statistics (`min`, `max`,
	`mean`, `std`). This matches retrieval code which expects two return values.
	"""
	vals = list(values)
	stats = _stats_from_vals(vals)
	if not vals:
		return ([], stats)

	if mode == NORMALIZATION_MIN_MAX:
		mn = stats["min"]
		mx = stats["max"]
		if mx - mn <= 1e-12:
			return ([0.0 for _ in vals], stats)
		normalized = [(v - mn) / (mx - mn) for v in vals]
		return (normalized, stats)

	if mode == NORMALIZATION_SOFTMAX:
		exps = [math.exp(v) for v in vals]
		s = sum(exps)
		if s == 0:
			return ([0.0 for _ in vals], stats)
		normalized = [e / s for e in exps]
		return (normalized, stats)

	if mode == NORMALIZATION_Z_SCORE:
		mean = stats["mean"]
		std = stats["std"] if stats["std"] > 0 else 1.0
		normalized = [(v - mean) / std for v in vals]
		return (normalized, stats)

	# fallback to identity
	return (list(vals), stats)

__all__ = ["normalize_scores", "NORMALIZATION_MIN_MAX", "NORMALIZATION_SOFTMAX", "NORMALIZATION_Z_SCORE"]
