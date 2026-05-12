"""Retrieval profiles and per-use-case runtime knobs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .rerank_presets import RerankWeights


@dataclass(frozen=True)
class RetrievalProfile:
    name: str
    candidate_k: int
    top_k: int
    rerank_with_cross_encoder: bool
    max_context_tokens: int
    normalization: str
    max_chunks_per_source: int
    semantic_weight: float
    bm25_weight: float
    title_weight: float
    company_weight: float
    skills_weight: float
    category_weight: float
    entity_bias_weight: float

    def weights(self) -> RerankWeights:
        return RerankWeights(
            semantic_weight=self.semantic_weight,
            bm25_weight=self.bm25_weight,
            title_weight=self.title_weight,
            company_weight=self.company_weight,
            skills_weight=self.skills_weight,
            category_weight=self.category_weight,
            entity_bias_weight=self.entity_bias_weight,
        ).normalize()


PROFILES: Dict[str, RetrievalProfile] = {
    "balanced": RetrievalProfile(
        name="balanced",
        candidate_k=28,
        top_k=8,
        rerank_with_cross_encoder=False,
        max_context_tokens=1800,
        normalization="min-max",
        max_chunks_per_source=2,
        semantic_weight=0.35,
        bm25_weight=0.28,
        title_weight=0.12,
        company_weight=0.10,
        skills_weight=0.10,
        category_weight=0.03,
        entity_bias_weight=0.02,
    ),
    "semantic-heavy": RetrievalProfile(
        name="semantic-heavy",
        candidate_k=30,
        top_k=8,
        rerank_with_cross_encoder=False,
        max_context_tokens=1700,
        normalization="z-score",
        max_chunks_per_source=2,
        semantic_weight=0.56,
        bm25_weight=0.16,
        title_weight=0.10,
        company_weight=0.06,
        skills_weight=0.08,
        category_weight=0.02,
        entity_bias_weight=0.02,
    ),
    "keyword-heavy": RetrievalProfile(
        name="keyword-heavy",
        candidate_k=34,
        top_k=8,
        rerank_with_cross_encoder=False,
        max_context_tokens=1700,
        normalization="min-max",
        max_chunks_per_source=2,
        semantic_weight=0.22,
        bm25_weight=0.50,
        title_weight=0.10,
        company_weight=0.06,
        skills_weight=0.09,
        category_weight=0.02,
        entity_bias_weight=0.01,
    ),
    "recommendation": RetrievalProfile(
        name="recommendation",
        candidate_k=40,
        top_k=10,
        rerank_with_cross_encoder=True,
        max_context_tokens=2100,
        normalization="softmax",
        max_chunks_per_source=2,
        semantic_weight=0.44,
        bm25_weight=0.20,
        title_weight=0.11,
        company_weight=0.08,
        skills_weight=0.11,
        category_weight=0.04,
        entity_bias_weight=0.02,
    ),
    "faq": RetrievalProfile(
        name="faq",
        candidate_k=20,
        top_k=6,
        rerank_with_cross_encoder=False,
        max_context_tokens=1200,
        normalization="softmax",
        max_chunks_per_source=1,
        semantic_weight=0.40,
        bm25_weight=0.35,
        title_weight=0.08,
        company_weight=0.07,
        skills_weight=0.07,
        category_weight=0.02,
        entity_bias_weight=0.01,
    ),
    "strict-job-search": RetrievalProfile(
        name="strict-job-search",
        candidate_k=36,
        top_k=8,
        rerank_with_cross_encoder=True,
        max_context_tokens=1900,
        normalization="z-score",
        max_chunks_per_source=1,
        semantic_weight=0.34,
        bm25_weight=0.34,
        title_weight=0.14,
        company_weight=0.08,
        skills_weight=0.08,
        category_weight=0.01,
        entity_bias_weight=0.01,
    ),
}


def resolve_profile(profile_name: str) -> RetrievalProfile:
    if not profile_name:
        return PROFILES["balanced"]
    return PROFILES.get(profile_name.lower(), PROFILES["balanced"])
