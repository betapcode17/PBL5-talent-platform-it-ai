"""Benchmark and evaluation helpers for the retrieval layer."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence

from .context_packing import estimate_tokens, pack_context
from .retrieval import RAGRetrievalService
from .retrieval_profiles import PROFILES


@dataclass(frozen=True)
class EvaluationCase:
    query: str
    expected_terms: Sequence[str]
    profile: str = "balanced"


def _quality_score(texts: Sequence[str], expected_terms: Sequence[str]) -> float:
    if not texts or not expected_terms:
        return 0.0
    corpus = " ".join(texts).lower()
    matches = sum(1 for term in expected_terms if term.lower() in corpus)
    return matches / max(1, len(expected_terms))


def benchmark_retrieval(service: RAGRetrievalService, cases: Sequence[EvaluationCase]) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    latencies: List[float] = []
    token_counts: List[int] = []
    quality_scores: List[float] = []
    grouped: Dict[str, List[Dict[str, object]]] = {}

    for case in cases:
        started = time.perf_counter()
        retrieval = service.retrieve(case.query, profile_name=case.profile)
        packed = pack_context(
            retrieval.items,
            max_context_tokens=PROFILES.get(case.profile, PROFILES["balanced"]).max_context_tokens,
            max_chunk_tokens=320,
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        texts = [item.text for item in retrieval.items]
        quality = _quality_score(texts, case.expected_terms)
        token_usage = sum(estimate_tokens(item.text) for item in packed.items)

        rows.append(
            {
                "query": case.query,
                "profile": case.profile,
                "retrieved": len(retrieval.items),
                "latencyMs": elapsed_ms,
                "packedTokens": token_usage,
                "quality": quality,
                "topScore": retrieval.items[0].rerank_score if retrieval.items else 0.0,
            }
        )
        grouped.setdefault(case.profile, []).append(rows[-1])
        latencies.append(elapsed_ms)
        token_counts.append(token_usage)
        quality_scores.append(quality)

    by_profile: Dict[str, Dict[str, float]] = {}
    for profile, profile_rows in grouped.items():
        profile_latencies = [float(row["latencyMs"]) for row in profile_rows] # type: ignore
        profile_tokens = [float(row["packedTokens"]) for row in profile_rows] # type: ignore
        profile_quality = [float(row["quality"]) for row in profile_rows] # type: ignore
        by_profile[profile] = {
            "cases": float(len(profile_rows)),
            "avgLatencyMs": round(statistics.mean(profile_latencies), 2) if profile_latencies else 0.0,
            "avgPackedTokens": round(statistics.mean(profile_tokens), 2) if profile_tokens else 0.0,
            "avgQuality": round(statistics.mean(profile_quality), 4) if profile_quality else 0.0,
        }

    return {
        "summary": {
            "cases": len(rows),
            "avgLatencyMs": round(statistics.mean(latencies), 2) if latencies else 0.0,
            "p95LatencyMs": round(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 2) if latencies else 0.0,
            "avgPackedTokens": round(statistics.mean(token_counts), 2) if token_counts else 0.0,
            "avgQuality": round(statistics.mean(quality_scores), 4) if quality_scores else 0.0,
        },
        "byProfile": by_profile,
        "rows": rows,
    }
