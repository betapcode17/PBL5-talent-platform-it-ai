"""Query-aware metadata boosting for hybrid retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Set


COMPANY_HINTS = {"company", "congty", "doanhnghiep", "employer", "firm", "startup"}
ROLE_HINTS = {"backend", "frontend", "fullstack", "intern", "internship", "developer", "engineer", "job", "role", "position", "vieclam"}
SKILL_HINTS = {
    "python", "java", "golang", "node", "nestjs", "fastapi", "django", "spring",
    "react", "reactjs", "reactnative", "react-native", "vue", "vuejs", "javascript", "typescript",
    "frontend", "frontends", "frontenddeveloper", "web", "html", "css", "nextjs", "next.js",
    "sql", "postgresql", "mongodb", "redis", "docker", "kubernetes",
}


@dataclass(frozen=True)
class MetadataBoostPlan:
    title_multiplier: float
    company_multiplier: float
    skills_multiplier: float


@dataclass(frozen=True)
class MetadataBoostScores:
    title: float
    company: float
    skills: float
    category: float


def extract_terms(text: str) -> List[str]:
    return [term for term in re.findall(r"\b\w+\b", text.lower(), flags=re.UNICODE) if len(term) >= 2]


def field_match_score(query_terms: Set[str], field_value: Any) -> float:
    if not field_value:
        return 0.0
    field_terms = set(extract_terms(str(field_value)))
    if not field_terms:
        return 0.0
    overlap = len(query_terms.intersection(field_terms))
    return min(1.0, overlap / max(1.0, len(field_terms)))


def build_boost_plan(query_terms: Set[str]) -> MetadataBoostPlan:
    has_company = bool(COMPANY_HINTS.intersection(query_terms))
    has_role = bool(ROLE_HINTS.intersection(query_terms))
    has_skill = bool(SKILL_HINTS.intersection(query_terms))

    title_multiplier = 1.0 + (0.55 if has_role else 0.0)
    company_multiplier = 1.0 + (0.80 if has_company else 0.0)
    skills_multiplier = 1.0 + (0.70 if has_skill else 0.0)

    return MetadataBoostPlan(
        title_multiplier=title_multiplier,
        company_multiplier=company_multiplier,
        skills_multiplier=skills_multiplier,
    )


def compute_metadata_scores(query_terms: Set[str], metadata: Dict[str, Any], plan: MetadataBoostPlan) -> MetadataBoostScores:
    base_title = field_match_score(query_terms, metadata.get("title"))
    base_company = field_match_score(query_terms, metadata.get("company"))
    base_skills = field_match_score(query_terms, metadata.get("skills"))
    base_category = field_match_score(query_terms, metadata.get("category"))

    return MetadataBoostScores(
        title=min(1.0, base_title * plan.title_multiplier),
        company=min(1.0, base_company * plan.company_multiplier),
        skills=min(1.0, base_skills * plan.skills_multiplier),
        category=base_category,
    )
