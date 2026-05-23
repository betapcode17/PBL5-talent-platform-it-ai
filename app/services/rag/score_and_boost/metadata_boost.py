"""Metadata-based boosting utilities for retrieval scoring.

This module computes simple token overlap, fuzzy matching, and recency boosts
to influence hybrid ranking when semantic signals are weak.
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, NamedTuple, Optional, Set

from ..utils.text_utils import normalize_query_text, tokenize_for_bm25


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

JOB_TYPE_ALIASES = {
	"fulltime": {"fulltime", "full time", "full-time", "toan thoi gian", "toanthoigian"},
	"parttime": {"parttime", "part time", "part-time", "ban thoi gian", "banthoigian"},
	"remote": {"remote", "wfh", "work from home", "lam tu xa"},
	"hybrid": {"hybrid", "mix", "mixed"},
	"onsite": {"onsite", "on site", "on-site", "tai cho", "tai van phong"},
}


class MetadataBoostScores(NamedTuple):
	title: float
	company: float
	description: float
	category: float
	location: float
	job_type: float
	salary: float
	recency: float


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


def _job_type_score(terms: Set[str], field_value: str) -> float:
	if not field_value:
		return 0.0
	field_norm = normalize_query_text(field_value)
	terms_norm = normalize_query_text(" ".join(sorted(terms)))
	overlap = _token_overlap_score(terms, field_value)
	for aliases in JOB_TYPE_ALIASES.values():
		term_match = any(alias in terms_norm for alias in aliases)
		field_match = any(alias in field_norm for alias in aliases)
		if term_match and field_match:
			return 1.0
		if term_match or field_match:
			overlap = max(overlap, 0.6)
	return overlap


def _parse_updated_at(updated_at: object) -> Optional[float]:
	if updated_at is None:
		return None
	if isinstance(updated_at, (int, float)):
		return float(updated_at)
	if isinstance(updated_at, str):
		value = updated_at.strip()
		if not value:
			return None
		try:
			return float(value)
		except Exception:
			pass
		try:
			normalized = value.replace("Z", "+00:00")
			parsed = datetime.fromisoformat(normalized)
			if parsed.tzinfo is None:
				parsed = parsed.replace(tzinfo=timezone.utc)
			return parsed.timestamp()
		except Exception:
			return None
	return None


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
	category = _token_overlap_score(terms, str(metadata.get("category") or ""))
	location = max(_token_overlap_score(terms, str(metadata.get("location") or "")), _fuzzy_ratio(str(metadata.get("location") or ""), " ".join(terms)))
	job_type = _job_type_score(terms, str(metadata.get("job_type") or metadata.get("work_type") or ""))
	salary = max(_token_overlap_score(terms, str(metadata.get("salary") or metadata.get("salary_range") or "")), _fuzzy_ratio(str(metadata.get("salary") or metadata.get("salary_range") or ""), " ".join(terms)))

	recency = _recency_boost(_parse_updated_at(metadata.get("updated_at")), now_ts)
	# combine with simple linear mixing
	title_score = title * 0.8 + recency * 0.2
	company_score = company * 0.75 + recency * 0.25
	description_score = description * 0.9 + recency * 0.1
	category_score = category * 0.9
	location_score = location * 0.9
	job_type_score = job_type * 0.85
	salary_score = salary * 0.9

	return MetadataBoostScores(
		title=title_score,
		company=company_score,
		description=description_score,
		category=category_score,
		location=location_score,
		job_type=job_type_score,
		salary=salary_score,
		recency=recency,
	)

__all__ = ["ROLE_HINTS", "SKILL_HINTS", "MetadataBoostPlan", "MetadataBoostScores", "extract_terms", "build_boost_plan", "compute_metadata_scores"]
