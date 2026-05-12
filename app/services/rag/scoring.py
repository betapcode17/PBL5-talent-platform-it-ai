"""Hybrid scoring composition helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoreComponents:
    semantic: float
    bm25: float
    title: float
    company: float
    skills: float
    category: float
    entity_bias: float
    fulltext: float


def compose_hybrid_score(weights: object, components: ScoreComponents, fulltext_weight: float) -> float:
    score = (
        components.semantic * float(getattr(weights, "semantic_weight", 0.0))
        + components.bm25 * float(getattr(weights, "bm25_weight", 0.0))
        + components.title * float(getattr(weights, "title_weight", 0.0))
        + components.company * float(getattr(weights, "company_weight", 0.0))
        + components.skills * float(getattr(weights, "skills_weight", 0.0))
        + components.category * float(getattr(weights, "category_weight", 0.0))
        + components.entity_bias * float(getattr(weights, "entity_bias_weight", 0.0))
    )
    score += components.fulltext * max(0.0, min(1.0, fulltext_weight))
    return max(0.0, min(1.0, score))
