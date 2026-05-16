"""Simple query rewriter to expand abbreviations and normalize common job-query patterns."""

from __future__ import annotations

from typing import List

from .text_utils import normalize_query_text


ABBREVIATIONS = {
	"fe": "frontend",
	"frontend dev": "frontend developer",
	"be": "backend",
	"backend dev": "backend developer",
	"sde": "software engineer",
	"swe": "software engineer",
}


def rewrite_query(query: str) -> str:
	if not query:
		return ""
	q = normalize_query_text(query)
	parts: List[str] = []
	for token in q.split():
		parts.append(ABBREVIATIONS.get(token, token))
	rewritten = " ".join(parts)
	return rewritten

__all__ = ["rewrite_query"]
