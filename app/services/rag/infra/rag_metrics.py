"""Lightweight metrics collector for RAG operations (in-memory)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class RagMetrics:
    counters: Dict[str, int] = field(default_factory=dict)
    timers: Dict[str, float] = field(default_factory=dict)

    def incr(self, key: str, amount: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + amount

    def start_timer(self, key: str) -> float:
        now = time.perf_counter()
        self.timers[f"{key}_start"] = now
        return now

    def stop_timer(self, key: str) -> float:
        now = time.perf_counter()
        start = self.timers.pop(f"{key}_start", None)
        if start is None:
            return 0.0
        elapsed = now - start
        self.timers[key] = self.timers.get(key, 0.0) + elapsed
        return elapsed

    def snapshot(self) -> Dict[str, float]:
        snap: Dict[str, float] = {k: float(v) for k, v in self.counters.items()}
        snap.update({k: float(v) for k, v in self.timers.items() if not k.endswith("_start")})
        return snap


@dataclass
class RetrievalMetrics:
    # simple, attribute-style metrics used by retrieval/pipeline
    empty_retrieval_count: int = 0
    duplicate_removed_count: int = 0
    fallback_trigger_count: int = 0
    hallucination_fallback_count: int = 0
    semantic_distribution_mean: float = 0.0
    bm25_distribution_mean: float = 0.0
    normalization_min: float = 0.0
    normalization_max: float = 0.0
    normalization_std: float = 0.0
    rerank_latency_ms: Optional[float] = None
    context_packing_latency_ms: Optional[float] = None

    def update_latency(self, total_ms: float, rerank_ms: float, pack_ms: float) -> None:
        self.rerank_latency_ms = rerank_ms
        self.context_packing_latency_ms = pack_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "empty_retrieval_count": self.empty_retrieval_count,
            "duplicate_removed_count": self.duplicate_removed_count,
            "fallback_trigger_count": self.fallback_trigger_count,
            "hallucination_fallback_count": self.hallucination_fallback_count,
            "semantic_distribution_mean": self.semantic_distribution_mean,
            "bm25_distribution_mean": self.bm25_distribution_mean,
            "normalization_min": self.normalization_min,
            "normalization_max": self.normalization_max,
            "normalization_std": self.normalization_std,
            "rerank_latency_ms": self.rerank_latency_ms,
            "context_packing_latency_ms": self.context_packing_latency_ms,
        }
