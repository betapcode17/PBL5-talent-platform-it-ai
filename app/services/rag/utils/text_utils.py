"""Text normalization and tokenization utilities used across RAG modules."""

from __future__ import annotations

import re
import unicodedata
from typing import List


def remove_accents(text: str) -> str:
	if not text:
		return ""
	nfkd = unicodedata.normalize("NFKD", text)
	return "".join([c for c in nfkd if not unicodedata.combining(c)])


def normalize_query_text(text: str) -> str:
	if not text:
		return ""
	s = remove_accents(text)
	s = s.lower()
	s = re.sub(r"[^0-9a-z\s,.-]", " ", s)
	s = re.sub(r"\s+", " ", s).strip()
	return s


def tokenize_for_bm25(text: str) -> List[str]:
	if not text:
		return []
	s = normalize_query_text(text)
	# split on whitespace and punctuation
	tokens = re.findall(r"[a-z0-9]+", s)
	return tokens

__all__ = ["remove_accents", "normalize_query_text", "tokenize_for_bm25"]
