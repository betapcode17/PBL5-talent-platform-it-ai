"""Retrieval reranking weight presets for different optimization strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal


@dataclass
class RerankWeights:
    """Weights for hybrid retrieval scoring formula.
    
    Formula:
        score = (
            semantic * semantic_weight +
            bm25 * bm25_weight +
            title_bonus * title_weight +
            company_bonus * company_weight +
            skills_bonus * skills_weight +
            category_bonus * category_weight +
            entity_bonus * entity_bias_weight
        )
    """
    semantic_weight: float
    bm25_weight: float
    title_weight: float
    company_weight: float
    skills_weight: float
    category_weight: float
    entity_bias_weight: float
    
    @property
    def total_weight(self) -> float:
        """Sum of all weights (should be close to 1.0)."""
        return sum([
            self.semantic_weight,
            self.bm25_weight,
            self.title_weight,
            self.company_weight,
            self.skills_weight,
            self.category_weight,
            self.entity_bias_weight,
        ])
    
    def normalize(self) -> RerankWeights:
        """Normalize weights to sum to 1.0."""
        total = self.total_weight
        if total == 0:
            return self
        return RerankWeights(
            semantic_weight=self.semantic_weight / total,
            bm25_weight=self.bm25_weight / total,
            title_weight=self.title_weight / total,
            company_weight=self.company_weight / total,
            skills_weight=self.skills_weight / total,
            category_weight=self.category_weight / total,
            entity_bias_weight=self.entity_bias_weight / total,
        )


# Preset 1: BALANCED - semantic + BM25 + metadata (RECOMMENDED)
BALANCED_WEIGHTS = RerankWeights(
    semantic_weight=0.35,      # Vector similarity
    bm25_weight=0.30,          # Keyword matching
    title_weight=0.12,         # Title metadata
    company_weight=0.08,       # Company metadata
    skills_weight=0.08,        # Skills metadata
    category_weight=0.05,      # Category metadata
    entity_bias_weight=0.02,   # Entity type bias
)

# Preset 2: SEMANTIC_FIRST - prioritize semantic similarity (for conceptual queries)
SEMANTIC_FIRST_WEIGHTS = RerankWeights(
    semantic_weight=0.50,      # Heavy semantic weight
    bm25_weight=0.15,          # Light keyword matching
    title_weight=0.12,
    company_weight=0.08,
    skills_weight=0.08,
    category_weight=0.05,
    entity_bias_weight=0.02,
)

# Preset 3: BM25_FIRST - prioritize keyword matching (for exact searches)
BM25_FIRST_WEIGHTS = RerankWeights(
    semantic_weight=0.20,      # Light semantic
    bm25_weight=0.50,          # Heavy BM25 weight
    title_weight=0.12,
    company_weight=0.08,
    skills_weight=0.08,
    category_weight=0.02,
    entity_bias_weight=0.00,
)

# Preset 4: HYBRID_STRICT - equal semantic + BM25, minimal metadata
HYBRID_STRICT_WEIGHTS = RerankWeights(
    semantic_weight=0.45,      # Strong semantic
    bm25_weight=0.45,          # Strong BM25
    title_weight=0.05,         # Minimal metadata
    company_weight=0.03,
    skills_weight=0.02,
    category_weight=0.00,
    entity_bias_weight=0.00,
)

PRESETS: Dict[str, RerankWeights] = {
    "balanced": BALANCED_WEIGHTS.normalize(),
    "semantic-first": SEMANTIC_FIRST_WEIGHTS.normalize(),
    "bm25-first": BM25_FIRST_WEIGHTS.normalize(),
    "hybrid": HYBRID_STRICT_WEIGHTS.normalize(),
}


def get_weights(preset_name: str) -> RerankWeights:
    """Get reranking weights by preset name."""
    weights = PRESETS.get(preset_name.lower(), BALANCED_WEIGHTS)
    return weights.normalize()
