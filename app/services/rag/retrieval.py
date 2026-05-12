"""Retrieval layer with semantic search, reranking, and source de-duplication."""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from .embedding import EmbeddingService
from .schemas import RetrievalResult, RetrievedChunk
from .settings import RAG_CACHE_TTL_SECONDS, RAG_CANDIDATE_K, RAG_MAX_CHUNKS_PER_SOURCE, RAG_TOP_K
from .vector_store import RAGVectorStore

logger = logging.getLogger(__name__)


class RAGRetrievalService:
    """Vector retrieval with lightweight metadata-aware reranking and TTL cache."""

    def __init__(self, vector_store: RAGVectorStore, embedding: EmbeddingService) -> None:
        self.vector_store = vector_store
        self.embedding = embedding
        self._cache: Dict[str, Tuple[float, RetrievalResult]] = {}

    def retrieve(self, query: str, filters: Optional[Dict[str, Any]] = None) -> RetrievalResult:
        start = time.perf_counter()
        cache_key = self._build_cache_key(query, filters)
        cached = self._cache.get(cache_key)
        now = time.time()
        if cached and now - cached[0] <= RAG_CACHE_TTL_SECONDS:
            cached_result = cached[1]
            return RetrievalResult(
                query=cached_result.query,
                items=cached_result.items,
                latency_ms=round((time.perf_counter() - start) * 1000, 2),
                cached=True,
            )

        query_emb = self.embedding.encode([query])[0]
        raw = self.vector_store.query(query_emb, top_k=RAG_CANDIDATE_K, where=filters)

        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        dists = (raw.get("distances") or [[]])[0]
        ids = (raw.get("ids") or [[]])[0]

        query_terms = self._extract_terms(query)
        reranked: List[RetrievedChunk] = []

        for idx, text in enumerate(docs):
            metadata = metas[idx] if idx < len(metas) else {}
            distance = float(dists[idx]) if idx < len(dists) else 1.0
            chunk_id = str(ids[idx]) if idx < len(ids) else f"chunk-{idx}"
            source_key = str(metadata.get("source_key") or metadata.get("source_id") or chunk_id)

            overlap = self._lexical_overlap(query_terms, self._extract_terms(text))
            title_bonus = self._field_match_bonus(query_terms, metadata.get("title"))
            company_bonus = self._field_match_bonus(query_terms, metadata.get("company"))
            skill_bonus = self._field_match_bonus(query_terms, metadata.get("skills"))
            semantic_score = max(0.0, 1.0 - distance)
            rerank_score = semantic_score * 0.65 + overlap * 0.2 + title_bonus * 0.1 + max(company_bonus, skill_bonus) * 0.05

            reranked.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    metadata=metadata,
                    distance=distance,
                    rerank_score=round(rerank_score, 6),
                    source_key=source_key,
                )
            )

        reranked.sort(key=lambda item: item.rerank_score, reverse=True)
        items = self._dedupe_sources(reranked)
        result = RetrievalResult(
            query=query,
            items=items,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            cached=False,
        )
        self._cache[cache_key] = (now, result)
        return result

    @staticmethod
    def _build_cache_key(query: str, filters: Optional[Dict[str, Any]]) -> str:
        return f"{query.lower().strip()}::{sorted((filters or {}).items())}"

    @staticmethod
    def _extract_terms(text: str) -> set[str]:
        return {term for term in re.findall(r"\b\w+\b", text.lower(), flags=re.UNICODE) if len(term) >= 2}

    @staticmethod
    def _lexical_overlap(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a.intersection(b)) / float(len(a))

    def _field_match_bonus(self, query_terms: set[str], field_value: Any) -> float:
        if not field_value:
            return 0.0
        field_terms = self._extract_terms(str(field_value))
        if not field_terms:
            return 0.0
        return min(1.0, len(query_terms.intersection(field_terms)) / max(1.0, len(field_terms)))

    @staticmethod
    def _dedupe_sources(items: List[RetrievedChunk]) -> List[RetrievedChunk]:
        selected: List[RetrievedChunk] = []
        per_source: Dict[str, int] = {}
        seen_texts: set[str] = set()

        for item in items:
            normalized_text = " ".join(item.text.split())
            if normalized_text in seen_texts:
                continue

            used = per_source.get(item.source_key, 0)
            if used >= RAG_MAX_CHUNKS_PER_SOURCE:
                continue

            selected.append(item)
            seen_texts.add(normalized_text)
            per_source[item.source_key] = used + 1

            if len(selected) >= RAG_TOP_K:
                break

        return selected
