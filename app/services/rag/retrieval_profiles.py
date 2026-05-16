"""Simple retrieval profiles and resolution helpers.

This module provides a minimal compatibility implementation for the
`RetrievalProfile` concept used across the pipeline. It intentionally keeps
profiles small and serializable so that the admin UI can enumerate them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class RetrievalProfile:
    name: str
    candidate_k: int
    top_k: int
    rerank_with_cross_encoder: bool
    max_context_tokens: int
    normalization: str
    max_chunks_per_source: int

    # weight attributes expected elsewhere
    semantic_weight: float = 1.0
    bm25_weight: float = 1.0
    title_weight: float = 0.5
    company_weight: float = 0.5
    description_weight: float = 0.5
    skills_weight: float = 0.5
    category_weight: float = 0.2
    entity_bias_weight: float = 0.1

    def weights(self) -> Dict[str, float]:
        return {
            "semantic": self.semantic_weight,
            "bm25": self.bm25_weight,
            "title": self.title_weight,
            "company": self.company_weight,
            "description": self.description_weight,
            "skills": self.skills_weight,
            "category": self.category_weight,
            "entity_bias": self.entity_bias_weight,
        }


# minimal set of profiles
PROFILES: Dict[str, RetrievalProfile] = {
    "balanced": RetrievalProfile(name="balanced", candidate_k=64, top_k=8, rerank_with_cross_encoder=False, max_context_tokens=1500, normalization="minmax", max_chunks_per_source=4),
    "strict-job-search": RetrievalProfile(name="strict-job-search", candidate_k=96, top_k=12, rerank_with_cross_encoder=True, max_context_tokens=2000, normalization="minmax", max_chunks_per_source=6, semantic_weight=1.2, bm25_weight=0.8),
    "faq": RetrievalProfile(name="faq", candidate_k=24, top_k=6, rerank_with_cross_encoder=False, max_context_tokens=800, normalization="minmax", max_chunks_per_source=2, semantic_weight=0.6, bm25_weight=1.4),
    "recommendation": RetrievalProfile(name="recommendation", candidate_k=48, top_k=6, rerank_with_cross_encoder=False, max_context_tokens=1200, normalization="minmax", max_chunks_per_source=4, semantic_weight=1.4, bm25_weight=0.6),
}


def resolve_profile(name: Optional[str]) -> RetrievalProfile:
    if not name:
        return PROFILES["balanced"]
    return PROFILES.get(str(name), PROFILES["balanced"])


def detect_and_resolve_profile(query: str, profile_name: Optional[str]) -> RetrievalProfile:
    # very small heuristic: delegate to explicit override when provided
    if profile_name:
        return resolve_profile(profile_name)
    q = (query or "").lower()
    if any(tok in q for tok in ("faq", "policy", "quydinh", "rule")):
        return PROFILES["faq"]
    if any(tok in q for tok in ("recommend", "goiy", "recommendation")):
        return PROFILES["recommendation"]
    if any(tok in q for tok in ("salary", "luong", "mieu ta", "diadiem", "remote")):
        return PROFILES["strict-job-search"]
    return PROFILES["balanced"]
