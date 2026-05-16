"""Scoring composition helpers for hybrid retrieval.

Expose utilities to combine semantic similarity, BM25, metadata boosts
and recency into a single sortable score used to rank candidates prior
to optional cross-encoder reranking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ScoreComponents:
	semantic: float = 0.0
	bm25: float = 0.0
	title: float = 0.0
	company: float = 0.0
	description: float = 0.0
	skills: float = 0.0
	category: float = 0.0
	recency: float = 0.0
	entity_bias: float = 0.0
	fulltext: float = 0.0


def compose_hybrid_score(arg1, arg2=None, fulltext_weight: float = 0.0) -> float:
	"""Compose a hybrid score from components and weights.

	Supports two call patterns for backward compatibility:
	- compose_hybrid_score(components, weights)
	- compose_hybrid_score(weights, components, fulltext_weight)
	"""
	# resolve args
	if isinstance(arg1, dict):
		weights = arg1
		components = arg2
	else:
		components = arg1
		weights = arg2 or {}

	if components is None:
		return 0.0

	# component multipliers
	s_sem = components.semantic * weights.get("semantic", 1.0)
	s_bm25 = components.bm25 * weights.get("bm25", 0.0)
	s_title = components.title * weights.get("title", 0.0)
	s_company = components.company * weights.get("company", 0.0)
	s_desc = components.description * weights.get("description", 0.0)
	s_skills = components.skills * weights.get("skills", 0.0)
	s_cat = components.category * weights.get("category", 0.0)
	s_recency = components.recency * weights.get("recency", 0.0)
	s_entity = components.entity_bias * weights.get("entity_bias", 0.0)
	# fulltext gets an explicit external weight if provided, otherwise from weights
	s_fulltext = components.fulltext * (fulltext_weight if fulltext_weight else weights.get("fulltext", 0.0))

	base = s_sem + s_bm25 + s_fulltext + s_entity
	meta = s_title + s_company + s_desc + s_skills + s_cat + s_recency
	# ensure some normalization so values stay in a comparable range
	final = base * (1.0 + 0.5 * meta)
	# clamp to a stable range
	return float(max(-1.0, min(1.0, final)))

__all__ = ["ScoreComponents", "compose_hybrid_score"]
