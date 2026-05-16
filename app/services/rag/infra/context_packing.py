"""Token-aware context packing relocated under infra."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from ..schemas_and_settings.schemas import RetrievedChunk


def estimate_tokens(text: str) -> int:
    word_count = max(1, len(text.split()))
    return int(word_count * 1.35)


@dataclass(frozen=True)
class PackedChunk:
    chunk_id: str
    source_key: str
    text: str
    metadata: Dict[str, Any]
    rerank_score: float
    tokens: int
    truncated: bool


@dataclass(frozen=True)
class ContextPackingResult:
    items: List[PackedChunk]
    total_tokens: int
    truncated_count: int
    dropped_count: int


def pack_context(
    items: Sequence[RetrievedChunk],
    max_context_tokens: int,
    max_chunk_tokens: int,
) -> ContextPackingResult:
    if max_context_tokens <= 0:
        return ContextPackingResult(items=[], total_tokens=0, truncated_count=0, dropped_count=len(items))

    sorted_items = sorted(items, key=lambda x: x.rerank_score, reverse=True)

    packed: List[PackedChunk] = []
    total_tokens = 0
    truncated_count = 0

    for item in sorted_items:
        remaining = max_context_tokens - total_tokens
        if remaining <= 0:
            break

        raw_text = item.text.strip()
        allowed_tokens = min(remaining, max_chunk_tokens)
        text_tokens = estimate_tokens(raw_text)

        if text_tokens <= allowed_tokens:
            packed_text = raw_text
            used_tokens = text_tokens
            truncated = False
        else:
            words = raw_text.split()
            max_words = max(12, int(allowed_tokens / 1.35))
            packed_text = " ".join(words[:max_words])
            used_tokens = estimate_tokens(packed_text)
            truncated = True

        if used_tokens <= 0 or not packed_text:
            continue

        packed.append(
            PackedChunk(
                chunk_id=item.chunk_id,
                source_key=item.source_key,
                text=packed_text,
                metadata=item.metadata,
                rerank_score=item.rerank_score,
                tokens=used_tokens,
                truncated=truncated,
            )
        )
        total_tokens += used_tokens
        if truncated:
            truncated_count += 1

    dropped_count = max(0, len(items) - len(packed))
    return ContextPackingResult(
        items=packed,
        total_tokens=total_tokens,
        truncated_count=truncated_count,
        dropped_count=dropped_count,
    )
