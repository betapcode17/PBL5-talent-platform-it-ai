"""Rerank presets relocated under rerankers package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class RerankWeights:
    semantic_weight: float
    bm25_weight: float
    title_weight: float
    company_weight: float
    description_weight: float
    skills_weight: float
    category_weight: float
    entity_bias_weight: float

    @property
    def total_weight(self) -> float:
        return sum([
            self.semantic_weight,
            self.bm25_weight,
            self.title_weight,
            self.company_weight,
            self.description_weight,
            self.skills_weight,
            self.category_weight,
            self.entity_bias_weight,
        ])

    def normalize(self) -> "RerankWeights":
        total = self.total_weight
        if total == 0:
            return self
        return RerankWeights(
            semantic_weight=self.semantic_weight / total,
            bm25_weight=self.bm25_weight / total,
            title_weight=self.title_weight / total,
            company_weight=self.company_weight / total,
            description_weight=self.description_weight / total,
            skills_weight=self.skills_weight / total,
            category_weight=self.category_weight / total,
            entity_bias_weight=self.entity_bias_weight / total,
        )


BALANCED_WEIGHTS = RerankWeights(0.35, 0.30, 0.10, 0.07, 0.07, 0.06, 0.03, 0.02)
SEMANTIC_FIRST_WEIGHTS = RerankWeights(0.5, 0.15, 0.10, 0.07, 0.07, 0.06, 0.03, 0.02)
BM25_FIRST_WEIGHTS = RerankWeights(0.2, 0.5, 0.10, 0.07, 0.07, 0.06, 0.0, 0.0)
HYBRID_STRICT_WEIGHTS = RerankWeights(0.45, 0.45, 0.04, 0.02, 0.02, 0.02, 0.0, 0.0)

PRESETS: Dict[str, RerankWeights] = {
    "balanced": BALANCED_WEIGHTS.normalize(),
    "semantic-first": SEMANTIC_FIRST_WEIGHTS.normalize(),
    "bm25-first": BM25_FIRST_WEIGHTS.normalize(),
    "hybrid": HYBRID_STRICT_WEIGHTS.normalize(),
}


def get_weights(preset_name: str) -> RerankWeights:
    return PRESETS.get(preset_name.lower(), BALANCED_WEIGHTS).normalize()
