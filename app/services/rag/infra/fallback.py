"""Helpers to detect and normalize fallback placeholders in backend payloads."""

from __future__ import annotations

from typing import Any


FALLBACK_MARKERS = {
    "Chua co thong tin ro rang",
    "CHUA_CO_THONG_TIN_RO_RANG",
    "N/A",
    "None",
}


def is_fallback_value(value: Any) -> bool:
    if value is None:
        return True
    try:
        s = str(value).strip()
    except Exception:
        return False
    if not s:
        return True
    return s in FALLBACK_MARKERS or s.lower().startswith("chua co")


def normalize_fallback(value: Any, replacement: str = "") -> str:
    if is_fallback_value(value):
        return replacement
    return str(value)


from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class LowConfidenceResult:
    low_confidence: bool
    safe_response: Optional[str]
    reason: str


def detect_low_confidence(items: List[Dict[str, Any]], min_top_score: float = 0.5, min_item_count: int = 2) -> LowConfidenceResult:
    """Very small heuristic to decide when retrieval is low-confidence."""
    if not items or len(items) < min_item_count:
        return LowConfidenceResult(low_confidence=True, safe_response=None, reason="insufficient_items")
    try:
        top_score = max(getattr(it, "rerank_score", 0.0) for it in items)
    except Exception:
        top_score = max((it.get("rerank_score", 0.0) if isinstance(it, dict) else 0.0) for it in items)
    if top_score < min_top_score:
        return LowConfidenceResult(low_confidence=True, safe_response=None, reason="low_top_score")
    return LowConfidenceResult(low_confidence=False, safe_response=None, reason="ok")


def fallback_observability_payload(stage: str, reason: str, enabled: bool = True) -> Dict[str, Any]:
    return {"triggered": bool(enabled and reason != "ok"), "stage": stage, "reason": reason}
