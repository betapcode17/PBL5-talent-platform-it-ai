"""Production-oriented hybrid retrieval with normalization, dedupe, fallback, and observability."""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rank_bm25 import BM25Okapi  # type: ignore

from .cross_encoder import CrossEncoderReranker
from .dedupe import DedupeStats, dedupe_and_diversify
from .embedding import EmbeddingService
from .fallback import detect_low_confidence, fallback_observability_payload
from .fulltext_store import RAGFullTextStore
from .metadata_boost import build_boost_plan, compute_metadata_scores, extract_terms
from .normalization import normalize_scores
from .rag_metrics import RetrievalMetrics
from .retrieval_profiles import RetrievalProfile, resolve_profile
from .schemas import RetrievalResult, RetrievedChunk
from .settings import (
    RAG_CACHE_TTL_SECONDS,
    RAG_DEFAULT_PROFILE,
    RAG_DEDUPE_SIMILARITY_THRESHOLD,
    RAG_ENABLE_BM25,
    RAG_ENABLE_FALLBACK,
    RAG_ENABLE_SEMANTIC,
    RAG_FULLTEXT_TOP_K,
    RAG_FULLTEXT_WEIGHT,
    RAG_LOW_CONFIDENCE_MIN_ITEMS,
    RAG_LOW_CONFIDENCE_TOP_SCORE,
    RAG_NORMALIZATION_STRATEGY,
    RAG_RETRIEVAL_RETRY_COUNT,
    RAG_RERANK_PRESET,
    RAG_TOP_K,
    RAG_USE_CROSS_ENCODER,
    RAG_USE_FULLTEXT_SEARCH,
    RAG_USE_CUSTOM_WEIGHTS,
)
from .scoring import ScoreComponents, compose_hybrid_score
from .vector_store import RAGVectorStore
from .metadata_boost import ROLE_HINTS, SKILL_HINTS

logger = logging.getLogger(__name__)

CITY_ALIASES = {
    "hcm": {"hcm", "tphcm", "tp.hcm", "hochiminh", "hochiminhcity", "ho-chi-minh", "ho chi minh", "ho chi minh city", "saigon"},
    "hanoi": {"hanoi", "ha noi"},
    "danang": {"danang", "da nang"},
    "cantho": {"cantho", "can tho"},
}

EDUCATION_HINTS = {"tutorial", "course", "learn", "guide", "example", "examples", "docs", "documentation"}
BACKEND_SKILL_HINTS = {"java", "python", "golang", "node", "nestjs", "fastapi", "django", "spring", "sql", "postgresql", "mongodb", "redis", "docker", "kubernetes"}
FRONTEND_SKILL_HINTS = {"react", "reactjs", "reactnative", "vue", "vuejs", "javascript", "typescript", "html", "css", "nextjs"}


class RAGRetrievalService:
    """Hybrid retrieval with semantic search, BM25, reranking, fallback, and metrics."""

    def __init__(self, vector_store: RAGVectorStore, embedding: EmbeddingService) -> None:
        self.vector_store = vector_store
        self.embedding = embedding
        self._cache: Dict[str, Tuple[float, RetrievalResult]] = {}
        self.fulltext_store = RAGFullTextStore() if RAG_USE_FULLTEXT_SEARCH else None
        self.cross_encoder = CrossEncoderReranker() if RAG_USE_CROSS_ENCODER else None
        self.metrics = RetrievalMetrics()
        self.last_profile = resolve_profile(RAG_DEFAULT_PROFILE)
        self.last_dedupe_stats = DedupeStats(0, 0, 0, 0, 0)
        self.last_fallback: Dict[str, Any] = fallback_observability_payload("none", "not-run", enabled=False)
        self.last_context_budget_tokens = self.last_profile.max_context_tokens
        logger.info(
            "rag.retrieval.initialized preset=%s custom=%s fulltext=%s cross_encoder=%s default_profile=%s",
            RAG_RERANK_PRESET,
            RAG_USE_CUSTOM_WEIGHTS,
            bool(self.fulltext_store),
            bool(self.cross_encoder and self.cross_encoder.available),
            self.last_profile.name,
        )

    def retrieve(self, query: str, filters: Optional[Dict[str, Any]] = None, profile_name: Optional[str] = None) -> RetrievalResult:
        started = time.perf_counter()
        cache_key = self._build_cache_key(query, filters, profile_name)
        cached = self._cache.get(cache_key)
        now = time.time()

        if cached and now - cached[0] <= RAG_CACHE_TTL_SECONDS:
            cached_result = cached[1]
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            logger.info(
                "rag.retrieval.complete cached=true returned=%s latency_ms=%s top_score=%s profile=%s query=%r",
                len(cached_result.items),
                latency_ms,
                cached_result.items[0].rerank_score if cached_result.items else None,
                self.last_profile.name,
                query[:120],
            )
            return RetrievalResult(query=cached_result.query, items=cached_result.items, latency_ms=latency_ms, cached=True)

        profile = self._resolve_runtime_profile(query, profile_name)
        self.last_profile = profile
        self.last_context_budget_tokens = profile.max_context_tokens

        candidates = self._collect_candidates_with_fallback(query, filters, profile)
        if not candidates:
            self.metrics.empty_retrieval_count += 1
            self.last_fallback = fallback_observability_payload("retrieval", "empty-candidates", enabled=True)
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            return RetrievalResult(query=query, items=[], latency_ms=latency_ms, cached=False)

        query_terms = extract_terms(query)
        query_term_set = set(query_terms)
        query_plan = self._build_job_query_plan(query, query_term_set)
        boost_plan = build_boost_plan(query_plan["terms"])
        city_hint = query_plan["city_hint"]
        effective_query = query_plan["query"]

        semantic_raw = [max(0.0, min(1.0, 1.0 - float(candidate.get("distance", 1.0)))) for candidate in candidates]
        bm25_raw = self._compute_bm25_scores(query_plan["terms"], candidates) if RAG_ENABLE_BM25 else [0.0 for _ in candidates]

        semantic_scores, semantic_stats = normalize_scores(semantic_raw, profile.normalization)
        bm25_scores, bm25_stats = normalize_scores(bm25_raw, profile.normalization)
        self.metrics.semantic_distribution_mean = semantic_stats["mean"]
        self.metrics.bm25_distribution_mean = bm25_stats["mean"]
        self.metrics.normalization_min = min(semantic_stats["min"], bm25_stats["min"])
        self.metrics.normalization_max = max(semantic_stats["max"], bm25_stats["max"])
        self.metrics.normalization_std = max(semantic_stats["std"], bm25_stats["std"])

        rerank_started = time.perf_counter()
        reranked: List[RetrievedChunk] = []
        for idx, candidate in enumerate(candidates):
            metadata = candidate.get("metadata") or {}
            metadata_scores = compute_metadata_scores(query_plan["terms"], metadata, boost_plan)
            fulltext_score = float(candidate.get("fulltext_score", 0.0))
            rerank_score = compose_hybrid_score(
                profile.weights(),
                ScoreComponents(
                    semantic=semantic_scores[idx] if idx < len(semantic_scores) else 0.0,
                    bm25=bm25_scores[idx] if idx < len(bm25_scores) else 0.0,
                    title=metadata_scores.title,
                    company=metadata_scores.company,
                    skills=metadata_scores.skills,
                    category=metadata_scores.category,
                    entity_bias=1.0 if self._detect_entity_bias(query_plan["terms"]) == metadata.get("entity_type") else 0.0,
                    fulltext=fulltext_score,
                ),
                RAG_FULLTEXT_WEIGHT,
            )

            reranked.append(
                RetrievedChunk(
                    chunk_id=str(candidate.get("chunk_id") or f"chunk-{idx}"),
                    text=str(candidate.get("text") or ""),
                    metadata=metadata,
                    distance=float(candidate.get("distance", 1.0)),
                    rerank_score=round(rerank_score, 6),
                    source_key=str(metadata.get("source_key") or metadata.get("source_id") or candidate.get("chunk_id") or f"chunk-{idx}"),
                )
            )

        reranked.sort(key=lambda item: item.rerank_score, reverse=True)

        if profile.rerank_with_cross_encoder and self.cross_encoder and self.cross_encoder.available:
            reranked = self.cross_encoder.rerank(effective_query, reranked, top_k=min(profile.candidate_k, len(reranked)))

        rerank_latency_ms = round((time.perf_counter() - rerank_started) * 1000, 2)
        if city_hint:
            reranked = [item for item in reranked if self._matches_city_hint(item.metadata, city_hint)]

        reranked = [item for item in reranked if self._has_minimum_relevance(item.metadata, query_term_set)]
        selected = dedupe_and_diversify(
            reranked,
            top_k=min(profile.top_k, RAG_TOP_K),
            max_chunks_per_source=profile.max_chunks_per_source,
            similarity_threshold=RAG_DEDUPE_SIMILARITY_THRESHOLD,
        )
        self.last_dedupe_stats = selected.stats
        self.metrics.duplicate_removed_count = (
            selected.stats.text_duplicates_removed
            + selected.stats.similarity_duplicates_removed
            + selected.stats.source_limit_removed
        )

        fallback = detect_low_confidence(
            selected.items,
            min_top_score=RAG_LOW_CONFIDENCE_TOP_SCORE,
            min_item_count=RAG_LOW_CONFIDENCE_MIN_ITEMS,
        )
        self.last_fallback = fallback_observability_payload("retrieval", fallback.reason, enabled=fallback.low_confidence)
        if RAG_ENABLE_FALLBACK and fallback.low_confidence:
            self.metrics.fallback_trigger_count += 1
            if not selected.items:
                self.metrics.hallucination_fallback_count += 1

        total_latency_ms = round((time.perf_counter() - started) * 1000, 2)
        self.metrics.update_latency(total_latency_ms, rerank_latency_ms, 0.0)

        logger.info(
            "rag.retrieval.complete cached=false profile=%s candidates=%s returned=%s latency_ms=%s rerank_latency_ms=%s top_score=%s bm25_mean=%s semantic_mean=%s duplicate_removed=%s fallback=%s normalization=%s",
            profile.name,
            len(candidates),
            len(selected.items),
            total_latency_ms,
            rerank_latency_ms,
            selected.items[0].rerank_score if selected.items else None,
            round(self.metrics.bm25_distribution_mean, 4),
            round(self.metrics.semantic_distribution_mean, 4),
            self.metrics.duplicate_removed_count,
            self.last_fallback,
            profile.normalization,
        )

        result = RetrievalResult(query=query, items=selected.items, latency_ms=total_latency_ms, cached=False)
        self._cache[cache_key] = (now, result)
        return result

    def _resolve_runtime_profile(self, query: str, requested_profile: Optional[str]) -> RetrievalProfile:
        def _with_runtime_normalization(profile: RetrievalProfile) -> RetrievalProfile:
            if not RAG_NORMALIZATION_STRATEGY or profile.normalization == RAG_NORMALIZATION_STRATEGY:
                return profile
            return replace(profile, normalization=RAG_NORMALIZATION_STRATEGY)

        if requested_profile:
            return _with_runtime_normalization(resolve_profile(requested_profile))

        terms = set(extract_terms(query))
        if {"faq", "policy", "quydinh", "rule"} & terms:
            return _with_runtime_normalization(resolve_profile("faq"))
        if {"recommend", "recommendation", "goiy"} & terms:
            return _with_runtime_normalization(resolve_profile("recommendation"))
        if {"salary", "location", "remote", "hybrid", "onsite"} & terms:
            return _with_runtime_normalization(resolve_profile("strict-job-search"))
        if self._looks_like_job_search(terms):
            return _with_runtime_normalization(resolve_profile("strict-job-search"))
        return _with_runtime_normalization(resolve_profile(RAG_DEFAULT_PROFILE))

    def _collect_candidates_with_fallback(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
        profile: RetrievalProfile,
    ) -> List[Dict[str, Any]]:
        last_error: Optional[Exception] = None
        for _ in range(max(1, RAG_RETRIEVAL_RETRY_COUNT + 1)):
            try:
                candidates = self._collect_candidates(query, filters, profile)
                if candidates:
                    return candidates
            except Exception as exc:
                last_error = exc
                logger.warning("rag.retrieval.collect_candidates.retry error=%s", exc)

        if last_error:
            logger.error("rag.retrieval.collect_candidates.failed error=%s", last_error)
        return []

    def _collect_candidates(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
        profile: RetrievalProfile,
    ) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}

        if RAG_ENABLE_SEMANTIC:
            try:
                query_emb = self.embedding.encode([query])[0]
                raw = self.vector_store.query(query_emb, top_k=profile.candidate_k, where=filters)
                docs = (raw.get("documents") or [[]])[0]
                metas = (raw.get("metadatas") or [[]])[0]
                dists = (raw.get("distances") or [[]])[0]
                ids = (raw.get("ids") or [[]])[0]
                for idx, text in enumerate(docs):
                    metadata = metas[idx] if idx < len(metas) else {}
                    chunk_id = str(ids[idx]) if idx < len(ids) else f"chunk-{idx}"
                    merged[chunk_id] = {
                        "chunk_id": chunk_id,
                        "text": str(text),
                        "metadata": metadata,
                        "distance": float(dists[idx]) if idx < len(dists) else 1.0,
                        "fulltext_score": 0.0,
                    }
            except Exception as exc:
                logger.warning("rag.retrieval.semantic_failed query=%r error=%s", query[:120], exc)

        if self.fulltext_store is not None:
            try:
                for row in self.fulltext_store.query(query, top_k=RAG_FULLTEXT_TOP_K):
                    chunk_id = str(row.get("chunk_id") or "")
                    if not chunk_id:
                        continue
                    row_metadata = row.get("metadata") or {}
                    if not self._matches_filters(row_metadata, filters):
                        continue
                    current = merged.get(chunk_id)
                    if current is None:
                        merged[chunk_id] = {
                            "chunk_id": chunk_id,
                            "text": str(row.get("text") or ""),
                            "metadata": row_metadata,
                            "distance": 1.0,
                            "fulltext_score": float(row.get("fulltext_score", 0.0)),
                        }
                    else:
                        current["fulltext_score"] = max(float(current.get("fulltext_score", 0.0)), float(row.get("fulltext_score", 0.0)))
            except Exception as exc:
                logger.warning("rag.retrieval.fulltext_failed query=%r error=%s", query[:120], exc)

        # Graceful fallback if one side is disabled or empty.
        if not merged and RAG_ENABLE_FALLBACK and self.fulltext_store is not None:
            for row in self.fulltext_store.query(query, top_k=RAG_FULLTEXT_TOP_K):
                chunk_id = str(row.get("chunk_id") or "")
                if not chunk_id:
                    continue
                merged[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": str(row.get("text") or ""),
                    "metadata": row.get("metadata") or {},
                    "distance": 1.0,
                    "fulltext_score": float(row.get("fulltext_score", 0.0)),
                }

        return list(merged.values())

    @staticmethod
    def _compute_bm25_scores(query_terms: Sequence[str], candidates: Sequence[Dict[str, Any]]) -> List[float]:
        if not query_terms or not candidates:
            return [0.0 for _ in candidates]
        corpus = [extract_terms(str(item.get("text") or "")) for item in candidates]
        try:
            bm25 = BM25Okapi(corpus)
            return [float(score) for score in bm25.get_scores(list(query_terms))]
        except Exception as exc:
            logger.warning("rag.retrieval.bm25_failed error=%s", exc)
            return [0.0 for _ in candidates]

    @staticmethod
    def _build_cache_key(query: str, filters: Optional[Dict[str, Any]], profile_name: Optional[str]) -> str:
        return f"{query.lower().strip()}::{sorted((filters or {}).items())}::{profile_name or RAG_DEFAULT_PROFILE}"

    @staticmethod
    def _detect_entity_bias(query_terms: set[str]) -> Optional[str]:
        if {"company", "congty", "doanhnghiep", "employer"} & query_terms:
            return "company"
        if {"job", "vieclam", "position", "role"} & query_terms:
            return "job"
        return None

    @staticmethod
    def _detect_city_hint(query: str) -> Optional[str]:
        normalized = " ".join(extract_terms(query))
        compact = normalized.replace(" ", "")
        for canonical, aliases in CITY_ALIASES.items():
            for alias in aliases:
                alias_compact = alias.replace(" ", "").replace(".", "").replace("-", "")
                if alias in normalized or alias_compact in compact:
                    return canonical
        return None

    @classmethod
    def _build_job_query_plan(cls, query: str, query_terms: set[str]) -> Dict[str, Any]:
        city_hint = cls._detect_city_hint(query)
        role_hint = cls._infer_role_hint(query_terms)
        expanded_terms = set(query_terms)

        if role_hint:
            expanded_terms.add(role_hint)

        if city_hint:
            expanded_terms.add(city_hint)

        normalized_query_terms = [term for term in extract_terms(query) if term not in EDUCATION_HINTS]
        if role_hint and role_hint not in normalized_query_terms:
            normalized_query_terms = [role_hint, *normalized_query_terms]

        return {
            "query": " ".join(normalized_query_terms).strip() or query,
            "terms": expanded_terms,
            "city_hint": city_hint,
            "role_hint": role_hint,
        }

    @staticmethod
    def _looks_like_job_search(query_terms: set[str]) -> bool:
        if not query_terms or query_terms & EDUCATION_HINTS:
            return False
        return bool((ROLE_HINTS | SKILL_HINTS).intersection(query_terms))

    @staticmethod
    def _infer_role_hint(query_terms: set[str]) -> Optional[str]:
        if {"frontend", "front-end", "front_end"} & query_terms:
            return "frontend"
        if {"backend", "back-end", "back_end"} & query_terms:
            return "backend"
        if {"fullstack", "full-stack", "full_stack"} & query_terms:
            return "fullstack"
        if BACKEND_SKILL_HINTS.intersection(query_terms):
            return "backend"
        if FRONTEND_SKILL_HINTS.intersection(query_terms):
            return "frontend"
        return None

    @staticmethod
    def _matches_city_hint(metadata: Dict[str, Any], city_hint: str) -> bool:
        aliases = CITY_ALIASES.get(city_hint, {city_hint})
        location_text = " ".join(
            str(value).strip().lower()
            for value in (
                metadata.get("location"),
                metadata.get("city"),
            )
            if str(value or "").strip()
        )
        if not location_text:
            return False

        compact_location = location_text.replace(" ", "").replace(".", "").replace("-", "")
        for alias in aliases:
            alias_compact = alias.replace(" ", "").replace(".", "").replace("-", "")
            if alias in location_text or alias_compact in compact_location:
                return True
        return False

    @staticmethod
    def _has_minimum_relevance(metadata: Dict[str, Any], query_terms: set[str]) -> bool:
        if not query_terms:
            return True

        field_terms = set()
        for key in ("title", "company", "skills", "category", "location", "city", "job_type", "work_type", "level"):
            value = metadata.get(key)
            if value:
                field_terms.update(extract_terms(str(value)))

        return len(query_terms.intersection(field_terms)) > 0

    @staticmethod
    def _matches_filters(metadata: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
        if not filters:
            return True
        for key, value in filters.items():
            if metadata.get(key) != value:
                return False
        return True
