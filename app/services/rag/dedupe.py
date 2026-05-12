"""Deduplication and diversity helpers for retrieval results."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

from .schemas import RetrievedChunk


@dataclass(frozen=True)
class DedupeStats:
    total_input: int
    total_output: int
    text_duplicates_removed: int
    similarity_duplicates_removed: int
    source_limit_removed: int


@dataclass(frozen=True)
class SelectedRetrieval:
    items: List[RetrievedChunk]
    stats: DedupeStats


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _text_to_tf(text: str) -> Dict[str, float]:
    words = [word for word in _normalize_text(text).split(" ") if len(word) >= 2]
    tf: Dict[str, float] = {}
    for word in words:
        tf[word] = tf.get(word, 0.0) + 1.0
    norm = math.sqrt(sum(value * value for value in tf.values()))
    if norm <= 1e-9:
        return tf
    return {key: value / norm for key, value in tf.items()}


def cosine_text_similarity(left: str, right: str) -> float:
    left_tf = _text_to_tf(left)
    right_tf = _text_to_tf(right)
    if not left_tf or not right_tf:
        return 0.0
    dot = 0.0
    for key, value in left_tf.items():
        dot += value * right_tf.get(key, 0.0)
    return max(0.0, min(1.0, dot))


def dedupe_and_diversify(
    ranked_items: Sequence[RetrievedChunk],
    top_k: int,
    max_chunks_per_source: int,
    similarity_threshold: float,
) -> SelectedRetrieval:
    selected: List[RetrievedChunk] = []
    seen_texts: set[str] = set()
    per_source: Dict[str, int] = {}

    text_duplicates_removed = 0
    similarity_duplicates_removed = 0
    source_limit_removed = 0

    for item in ranked_items:
        if len(selected) >= top_k:
            break

        normalized_text = _normalize_text(item.text)
        if normalized_text in seen_texts:
            text_duplicates_removed += 1
            continue

        used = per_source.get(item.source_key, 0)
        if used >= max_chunks_per_source:
            source_limit_removed += 1
            continue

        is_near_duplicate = False
        for existing in selected:
            similarity = cosine_text_similarity(existing.text, item.text)
            if similarity >= similarity_threshold:
                is_near_duplicate = True
                break

        if is_near_duplicate:
            similarity_duplicates_removed += 1
            continue

        selected.append(item)
        seen_texts.add(normalized_text)
        per_source[item.source_key] = used + 1

    return SelectedRetrieval(
        items=selected,
        stats=DedupeStats(
            total_input=len(ranked_items),
            total_output=len(selected),
            text_duplicates_removed=text_duplicates_removed,
            similarity_duplicates_removed=similarity_duplicates_removed,
            source_limit_removed=source_limit_removed,
        ),
    )
