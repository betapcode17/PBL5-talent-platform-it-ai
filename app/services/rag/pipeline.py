"""End-to-end RAG pipeline orchestration service."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from app.services.backend_api_client import get_backend_client

from .embedding import EmbeddingService
from .generation import QwenGenerationService
from .ingestion import RAGIngestionService
from .retrieval import RAGRetrievalService
from .settings import RAG_SYNC_INTERVAL_SECONDS
from .vector_store import RAGVectorStore

logger = logging.getLogger(__name__)


class RAGPipelineService:
    """Coordinates ingestion, retrieval, and generation for chat requests."""

    def __init__(self) -> None:
        self.vector_store = RAGVectorStore()
        self.embedding = EmbeddingService()
        self.retrieval = RAGRetrievalService(self.vector_store, self.embedding)
        self.generation = QwenGenerationService()
        self.ingestion = RAGIngestionService(self.vector_store, self.embedding)
        self.backend = get_backend_client()
        self._sync_lock = asyncio.Lock()
        self._last_sync_at: float = 0.0
        self._sync_task: Optional[asyncio.Task] = None
        self._last_sync_result: Dict[str, Any] = {}

    async def ensure_index_fresh(self, force: bool = False) -> Dict[str, Any]:
        import torch  # type: ignore

        now = time.time()
        if not force and now - self._last_sync_at < RAG_SYNC_INTERVAL_SECONDS:
            return {"status": "skipped", "reason": "fresh", "lastSyncAt": self._last_sync_at, **self._last_sync_result}

        if not force and self.vector_store.count() > 0:
            self._schedule_background_sync()
            return {"status": "scheduled", "reason": "stale", "lastSyncAt": self._last_sync_at, **self._last_sync_result}

        async with self._sync_lock:
            now = time.time()
            if not force and now - self._last_sync_at < RAG_SYNC_INTERVAL_SECONDS:
                return {"status": "skipped", "reason": "fresh", "lastSyncAt": self._last_sync_at, **self._last_sync_result}

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            sync_result = await self.ingestion.sync_from_backend()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self._last_sync_at = time.time()
            self._last_sync_result = {
                "fetchedJobs": sync_result.fetched_jobs,
                "fetchedCompanies": sync_result.fetched_companies,
                "indexedChunks": sync_result.indexed_chunks,
                "indexedJobChunks": sync_result.indexed_job_chunks,
                "indexedCompanyChunks": sync_result.indexed_company_chunks,
                "backendPages": sync_result.backend_pages,
                "staleChunksRemoved": sync_result.stale_chunks_removed,
                "tookSeconds": sync_result.took_seconds,
                "errors": sync_result.errors or [],
                "warnings": sync_result.warnings,
            }
            return {"status": "ok", "lastSyncAt": self._last_sync_at, **self._last_sync_result}

    def _schedule_background_sync(self) -> None:
        if self._sync_task and not self._sync_task.done():
            return

        async def _runner() -> None:
            try:
                await self.ensure_index_fresh(force=True)
            except Exception as exc:
                logger.exception("Background sync failed: %s", exc)

        self._sync_task = asyncio.create_task(_runner())

    async def answer(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        filters: Optional[Dict[str, Any]] = None,
        extra_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        # Query-time calls should stay read-only and avoid a backend sync.
        # The index is refreshed explicitly through the reload/sync endpoints.
        sync_info = {"status": "skipped", "reason": "auto_sync_disabled"}
        retrieval = self.retrieval.retrieve(query, filters=filters)
        prompt = self._build_prompt(
            query=query,
            items=retrieval.items,
            history=conversation_history or [],
            extra_context=extra_context,
        )

        generation = await asyncio.to_thread(self.generation.generate, prompt)
        total_latency_ms = round((time.perf_counter() - started) * 1000, 2)

        sources = [
            {
                "entityType": item.metadata.get("entity_type"),
                "sourceId": item.metadata.get("source_id"),
                "id": item.metadata.get("job_id") or item.metadata.get("company_id"),
                "title": item.metadata.get("title"),
                "company": item.metadata.get("company"),
                "location": item.metadata.get("location"),
                "salary": item.metadata.get("salary"),
                "skills": item.metadata.get("skills"),
                "industry": item.metadata.get("industry"),
                "url": item.metadata.get("url"),
                "distance": item.distance,
                "score": item.rerank_score,
                "chunkId": item.chunk_id,
            }
            for item in retrieval.items
        ]

        return {
            "bot_response": generation.answer,
            "sources": sources,
            "detected_intent": "jobs_rag",
            "sync": sync_info,
            "generation": {
                "model": generation.model,
                "usedCuda": generation.used_cuda,
                "dtype": generation.dtype,
                "promptTokens": generation.prompt_tokens_estimate,
                "completionTokens": generation.completion_tokens_estimate,
            },
            "retrieval": {
                "count": len(retrieval.items),
                "latencyMs": retrieval.latency_ms,
                "cached": retrieval.cached,
            },
            "latencyMs": total_latency_ms,
            "promptPreview": prompt[:2000],
        }

    def health(self) -> Dict[str, Any]:
        import torch # type: ignore

        stats = self.vector_store.stats()
        cuda_info: Dict[str, Any] = {"available": torch.cuda.is_available()}
        if torch.cuda.is_available():
            cuda_info.update(
                {
                    "device": torch.cuda.get_device_name(0),
                    "allocatedBytes": torch.cuda.memory_allocated(0),
                    "reservedBytes": torch.cuda.memory_reserved(0),
                    "bf16Supported": torch.cuda.is_bf16_supported(),
                }
            )
        return {
            "vectorStore": stats,
            "cuda": cuda_info,
            "generationDevice": self.generation.device,
            "generationDtype": str(self.generation.dtype),
            "embedDevice": self.embedding.device,
            "lastSyncAt": self._last_sync_at,
            "lastSync": self._last_sync_result,
        }

    async def retrieval_debug(self, query: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        retrieval = self.retrieval.retrieve(query, filters=filters)
        return {
            "query": query,
            "count": len(retrieval.items),
            "latencyMs": retrieval.latency_ms,
            "cached": retrieval.cached,
            "items": [
                {
                    "chunkId": item.chunk_id,
                    "sourceKey": item.source_key,
                    "score": item.rerank_score,
                    "distance": item.distance,
                    "metadata": item.metadata,
                    "preview": item.text[:500],
                }
                for item in retrieval.items
            ],
        }

    async def list_indexed_jobs(self, page: int = 1, limit: int = 20) -> Dict[str, Any]:
        page = max(1, page)
        limit = max(1, min(limit, 100))

        # Read a larger slice than requested because one job may have multiple chunks.
        raw = self.vector_store.get(limit=min(limit * 6, 600), offset=max(0, (page - 1) * limit * 2))
        metadatas = raw.get("metadatas") or []

        deduped: List[Dict[str, Any]] = []
        seen_job_ids: set[str] = set()

        for metadata in metadatas:
            if not metadata or metadata.get("entity_type") != "job":
                continue

            job_id = str(metadata.get("job_id") or metadata.get("source_id") or "")
            if not job_id or job_id in seen_job_ids:
                continue

            seen_job_ids.add(job_id)
            deduped.append(
                {
                    "jobId": job_id,
                    "title": metadata.get("title"),
                    "company": metadata.get("company"),
                    "location": metadata.get("location"),
                    "salary": metadata.get("salary"),
                    "skills": metadata.get("skills"),
                    "category": metadata.get("category"),
                    "workType": metadata.get("work_type"),
                    "jobType": metadata.get("job_type"),
                    "level": metadata.get("level"),
                    "url": metadata.get("url"),
                    "updatedAt": metadata.get("updated_at"),
                }
            )

            if len(deduped) >= limit:
                break

        return {
            "page": page,
            "limit": limit,
            "count": len(deduped),
            "items": deduped,
            "vectorStoreCount": self.vector_store.count(),
        }

    @staticmethod
    def _build_prompt(
        query: str,
        items: List[Any],
        history: List[Dict[str, str]],
        extra_context: Optional[str] = None,
    ) -> str:
        history_text = ""
        if history:
            lines: List[str] = ["Recent conversation:"]
            for msg in history[-6:]:
                lines.append(f"{msg.get('role', 'user')}: {msg.get('content', '')}")
            history_text = "\n".join(lines)

        context_blocks: List[str] = []
        for i, item in enumerate(items, start=1):
            m = item.metadata
            entity_type = m.get("entity_type")
            if entity_type == "company":
                context_blocks.append(
                    (
                        f"[{i}] COMPANY\n"
                        f"Company ID: {m.get('company_id')}\n"
                        f"Company: {m.get('company')}\n"
                        f"Industry: {m.get('industry')}\n"
                        f"Location: {m.get('location')}\n"
                        f"Skills: {m.get('skills')}\n"
                        f"Website: {m.get('url')}\n"
                        f"Context: {item.text[:500]}\n"
                    )
                )
            else:
                context_blocks.append(
                    (
                        f"[{i}] JOB\n"
                        f"Job ID: {m.get('job_id')}\n"
                        f"Title: {m.get('title')}\n"
                        f"Company: {m.get('company')}\n"
                        f"Location: {m.get('location')}\n"
                        f"Salary: {m.get('salary')}\n"
                        f"Skills: {m.get('skills')}\n"
                        f"Category: {m.get('category')}\n"
                        f"URL: {m.get('url')}\n"
                        f"Context: {item.text[:500]}\n"
                    )
                )

        context_text = "\n---\n".join(context_blocks) if context_blocks else "No relevant backend context found."
        attachment_text = f"\nCandidate attachment context:\n{extra_context}\n" if extra_context else ""

        return (
            "You are a retrieval-grounded assistant for an IT hiring platform in Vietnam.\n"
            "Hard rules:\n"
            "1. Use only the retrieved context and attachment context.\n"
            "2. Do not invent salary, location, company facts, benefits, or requirements.\n"
            "3. If the context is missing or ambiguous, say that clearly.\n"
            "4. Prefer citing specific job titles or company names from the context.\n"
            "5. If multiple results are relevant, compare them briefly.\n"
            "6. Answer in Vietnamese.\n\n"
            f"{history_text}\n"
            f"{attachment_text}\n"
            f"Retrieved context:\n{context_text}\n\n"
            f"User query: {query}\n\n"
            "Write a helpful answer with:\n"
            "- short direct answer\n"
            "- relevant jobs or companies from context\n"
            "- concrete next step for the user\n"
        )


_pipeline_singleton: Optional[RAGPipelineService] = None


def get_rag_pipeline() -> RAGPipelineService:
    global _pipeline_singleton
    if _pipeline_singleton is None:
        _pipeline_singleton = RAGPipelineService()
    return _pipeline_singleton
