"""In-memory retrieval metrics for observability and Prometheus export."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class RetrievalMetrics:
    retrieval_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    context_packing_latency_ms: float = 0.0
    retrieval_count: int = 0
    empty_retrieval_count: int = 0
    duplicate_removed_count: int = 0
    fallback_trigger_count: int = 0
    hallucination_fallback_count: int = 0
    bm25_distribution_mean: float = 0.0
    semantic_distribution_mean: float = 0.0
    normalization_min: float = 0.0
    normalization_max: float = 0.0
    normalization_std: float = 0.0
    updated_at: float = field(default_factory=time.time)

    def update_latency(self, retrieval_ms: float, rerank_ms: float, packing_ms: float) -> None:
        self.retrieval_latency_ms = retrieval_ms
        self.rerank_latency_ms = rerank_ms
        self.context_packing_latency_ms = packing_ms
        self.retrieval_count += 1
        self.updated_at = time.time()

    def to_dict(self) -> Dict[str, float]:
        return {
            "retrievalLatencyMs": self.retrieval_latency_ms,
            "rerankLatencyMs": self.rerank_latency_ms,
            "contextPackingLatencyMs": self.context_packing_latency_ms,
            "retrievalCount": float(self.retrieval_count),
            "emptyRetrievalCount": float(self.empty_retrieval_count),
            "duplicateRemovedCount": float(self.duplicate_removed_count),
            "fallbackTriggerCount": float(self.fallback_trigger_count),
            "hallucinationFallbackCount": float(self.hallucination_fallback_count),
            "bm25DistributionMean": self.bm25_distribution_mean,
            "semanticDistributionMean": self.semantic_distribution_mean,
            "normalizationMin": self.normalization_min,
            "normalizationMax": self.normalization_max,
            "normalizationStd": self.normalization_std,
            "updatedAt": self.updated_at,
        }
