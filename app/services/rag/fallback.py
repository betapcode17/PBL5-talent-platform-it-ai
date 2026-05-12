"""Fallback strategy helpers for robust retrieval behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .schemas import RetrievedChunk


@dataclass(frozen=True)
class FallbackDecision:
    low_confidence: bool
    reason: str
    safe_response: str | None


def detect_low_confidence(items: List[RetrievedChunk], min_top_score: float, min_item_count: int) -> FallbackDecision:
    if not items:
        return FallbackDecision(
            low_confidence=True,
            reason="empty-retrieval",
            safe_response="Khong tim thay du lieu lien quan trong he thong hien tai.",
        )

    top_score = items[0].rerank_score
    if len(items) < min_item_count and top_score < min_top_score:
        return FallbackDecision(
            low_confidence=True,
            reason="weak-retrieval",
            safe_response="Du lieu truy xuat hien tai chua du manh de tra loi chinh xac. Vui long bo sung ky nang, cong ty hoac vi tri cu the hon.",
        )

    if top_score < min_top_score:
        return FallbackDecision(
            low_confidence=True,
            reason="low-top-score",
            safe_response="Khong tim thay du lieu lien quan ro rang. Vui long thu lai voi tu khoa cu the hon.",
        )

    return FallbackDecision(low_confidence=False, reason="ok", safe_response=None)


def fallback_observability_payload(stage: str, reason: str, enabled: bool = True) -> Dict[str, Any]:
    return {
        "triggered": bool(enabled),
        "stage": stage,
        "reason": reason,
    }
