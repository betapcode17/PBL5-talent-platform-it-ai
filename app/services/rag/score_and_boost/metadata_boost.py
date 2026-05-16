"""Metadata-based boosting utilities for retrieval scoring.

This module computes simple token overlap, fuzzy matching, and recency boosts
to influence hybrid ranking when semantic signals are weak.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, NamedTuple, Optional, Set

from ..utils.text_utils import tokenize_for_bm25


ROLE_HINTS = {"backend", "frontend", "fullstack", "data", "devops", "mobile", "qa", "product"}
SKILL_HINTS = {
	"java",
	"python",
	"javascript",
	"typescript",
	"react",
	"nextjs",
	"node",
	"nestjs",
	"spring",
	"django",
	"fastapi",
	"sql",
	"postgresql",
	"mongodb",
	"redis",
	"docker",
	"kubernetes",
}


class MetadataBoostScores(NamedTuple):
	title: float
	company: float
	description: float
	skills: float
	category: float


@dataclass
class MetadataBoostPlan:
	terms: Set[str]
	now_ts: float = time.time()


def extract_terms(text: str) -> List[str]:
	return tokenize_for_bm25(text)


def _token_overlap_score(terms: Set[str], field_value: Optional[str]) -> float:
	if not field_value or not terms:
		return 0.0
	tokens = set(tokenize_for_bm25(field_value))
	if not tokens:
		return 0.0
	return len(terms & tokens) / max(1, len(terms | tokens))


def _fuzzy_ratio(a: str, b: str) -> float:
	if not a or not b:
		return 0.0
	return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _recency_boost(updated_at: Optional[float], now_ts: float) -> float:
	if not updated_at:
		return 0.0
	try:
		days = max(0.0, (now_ts - float(updated_at)) / 86400.0)
		return 1.0 / (1.0 + math.log1p(days + 1.0))
	except Exception:
		return 0.0


def build_boost_plan(terms: Iterable[str]) -> MetadataBoostPlan:
	return MetadataBoostPlan(terms=set(t.lower() for t in terms if t))


def compute_metadata_scores(terms: Set[str], metadata: Dict[str, object], plan: MetadataBoostPlan) -> MetadataBoostScores:
	now_ts = plan.now_ts
	title = max(_token_overlap_score(terms, str(metadata.get("title") or "")), _fuzzy_ratio(str(metadata.get("title") or ""), " ".join(terms)))
	company = max(_token_overlap_score(terms, str(metadata.get("company") or "")), _fuzzy_ratio(str(metadata.get("company") or ""), " ".join(terms)))
	description = _token_overlap_score(terms, str(metadata.get("description") or ""))
	skills = _token_overlap_score(terms, ",".join(metadata.get("skills") or [])) # type: ignore
	category = _token_overlap_score(terms, str(metadata.get("category") or ""))

	recency = _recency_boost(metadata.get("updated_at"), now_ts) # type: ignore
	# combine with simple linear mixing
	title_score = title * 0.8 + recency * 0.2
	company_score = company * 0.75 + recency * 0.25
	description_score = description * 0.9 + recency * 0.1
	skills_score = skills * 0.85 + recency * 0.15
	category_score = category * 0.9

	return MetadataBoostScores(title=title_score, company=company_score, description=description_score, skills=skills_score, category=category_score)

__all__ = ["ROLE_HINTS", "SKILL_HINTS", "MetadataBoostPlan", "MetadataBoostScores", "extract_terms", "build_boost_plan", "compute_metadata_scores"]
