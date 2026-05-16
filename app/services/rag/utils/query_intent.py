"""Lightweight intent detector for simple retrieval profile selection."""

from __future__ import annotations

from typing import Optional

from .text_utils import tokenize_for_bm25


def detect_retrieval_profile(query: str) -> Optional[str]:
	if not query:
		return None
	tokens = tokenize_for_bm25(query)
	tset = set(tokens)
	if "senior" in tset or "lead" in tset or "manager" in tset:
		return "senior"
	if "intern" in tset or "internship" in tset:
		return "junior"
	if "remote" in tset or "work from home" in query:
		return "remote"
	return None

__all__ = ["detect_retrieval_profile"]
