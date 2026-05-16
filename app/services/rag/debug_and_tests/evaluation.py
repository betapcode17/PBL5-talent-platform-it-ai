"""Lightweight evaluation helpers used by the optional benchmark endpoint.

This file provides a minimal `EvaluationCase` and `benchmark_retrieval` so the
pipeline's `benchmark_retrieval` endpoint can function even when a richer test
harness is not present.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class EvaluationCase:
    query: str
    expected_terms: List[str]
    profile: str = "balanced"


def benchmark_retrieval(retrieval_service: Any, cases: List[EvaluationCase]) -> Dict[str, Any]:
    results = []
    for case in cases:
        try:
            res = retrieval_service.retrieve(case.query, profile_name=case.profile)
            results.append({"query": case.query, "count": len(res.items), "topScore": (res.items[0].rerank_score if res.items else None)})
        except Exception as exc:
            results.append({"query": case.query, "error": str(exc)})
    summary = {"cases": len(cases), "results": results}
    return summary
