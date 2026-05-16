"""End-to-end RAG pipeline orchestration service (moved into core).

Imports adjusted to reference stores and sibling core modules after reorganization.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..infra.context_packing import pack_context
from ..infra.embedding import EmbeddingService
try:
	from ..debug_and_tests.evaluation import EvaluationCase, benchmark_retrieval
except Exception:  # pragma: no cover - optional test harness
	EvaluationCase = None
	benchmark_retrieval = None
from ..infra.fallback import detect_low_confidence
from ..infra.generation import QwenGenerationService
from .ingestion import IngestionService
from ..prompt_and_response.prompt_builder import build_chat_prompt
from ..prompt_and_response.response_formatter import build_structured_chat_response, format_chat_response
from .retrieval import RAGRetrievalService
from ..retrieval_profiles import PROFILES
from ..schemas_and_settings.schemas import RetrievalResult, RetrievedChunk
from ..schemas_and_settings.settings import QWEN_MODEL_NAME, RAG_DEFAULT_PROFILE, RAG_MAX_CHUNK_TOKENS, RAG_RETRIEVAL_TIMEOUT_MS, RAG_SYNC_INTERVAL_SECONDS
from ..stores.vector_store import RAGVectorStore

logger = logging.getLogger(__name__)


def _public_skill_list(value: object) -> List[str]:
	if isinstance(value, list):
		skills = [str(item).strip() for item in value if str(item).strip()]
	else:
		skills = [segment.strip() for segment in str(value or "").split(",") if segment.strip()]

	return [skill for skill in skills if skill.lower() not in {"general", "not specified"}]


def _extract_document_section(text: str, start_label: str, end_labels: List[str]) -> str:
	if not text:
		return ""

	start_index = text.find(start_label)
	if start_index < 0:
		return ""

	section_start = start_index + len(start_label)
	section_end = len(text)
	for end_label in end_labels:
		end_index = text.find(end_label, section_start)
		if end_index >= 0:
			section_end = min(section_end, end_index)

	return text[section_start:section_end].strip()


def _public_description(value: object, text: str, start_label: str, end_labels: List[str]) -> str:
	description = str(value or "").strip()
	if description:
		return description
	return _extract_document_section(text, start_label, end_labels)


class RAGPipelineService:
	"""Coordinates independent ingest and chat paths for the RAG system."""

	def __init__(self) -> None:
		self.vector_store = RAGVectorStore()
		self.embedding = EmbeddingService()
		self.retrieval = RAGRetrievalService(self.vector_store, self.embedding)
		self.generation = QwenGenerationService()
		self.ingestion = IngestionService(self.vector_store, self.embedding)
		self._sync_lock = asyncio.Lock()
		self._last_sync_at: float = 0.0
		self._sync_task: Optional[asyncio.Task] = None
		self._last_sync_result: Dict[str, Any] = {}
		self._sync_status: str = "idle"
		self._last_sync_error: Optional[str] = None
		self._metrics: Dict[str, Any] = {
			"chatRequests": 0,
			"syncRequests": 0,
			"lastRetrievalLatencyMs": None,
			"lastGenerationLatencyMs": None,
			"lastPipelineLatencyMs": None,
			"lastRetrievedChunks": 0,
			"lastTopScore": None,
		}

	async def ensure_index_fresh(self, force: bool = False) -> Dict[str, Any]:
		import torch  # type: ignore

		now = time.time()
		if not force and now - self._last_sync_at < RAG_SYNC_INTERVAL_SECONDS:
			return {"status": "skipped", "reason": "fresh", "lastSyncAt": self._last_sync_at, "syncState": self._sync_status, **self._last_sync_result}

		async with self._sync_lock:
			self._metrics["syncRequests"] += 1
			now = time.time()
			if not force and now - self._last_sync_at < RAG_SYNC_INTERVAL_SECONDS:
				return {"status": "skipped", "reason": "fresh", "lastSyncAt": self._last_sync_at, "syncState": self._sync_status, **self._last_sync_result}

			self._sync_status = "running"
			if torch.cuda.is_available():
				torch.cuda.empty_cache()

			started = time.perf_counter()
			try:
				sync_result = await self.ingestion.sync_from_backend()  # type: ignore
			except Exception as exc:
				self._sync_status = "failed"
				self._last_sync_error = str(exc)
				logger.exception("rag.sync.failed error=%s", exc)
				raise
			finally:
				if torch.cuda.is_available():
					torch.cuda.empty_cache()

			latency_ms = round((time.perf_counter() - started) * 1000, 2)
			self._last_sync_at = time.time()
			self._sync_status = "idle"
			self._last_sync_error = None
			self._last_sync_result = {
				"fetchedJobs": sync_result.fetched_jobs,
				"fetchedCompanies": sync_result.fetched_companies,
				"indexedChunks": sync_result.indexed_chunks,
				"indexedJobChunks": sync_result.indexed_job_chunks,
				"indexedCompanyChunks": sync_result.indexed_company_chunks,
				"backendPages": sync_result.backend_pages,
				"staleChunksRemoved": sync_result.stale_chunks_removed,
				"tookSeconds": sync_result.took_seconds,
				"latencyMs": latency_ms,
				"errors": sync_result.errors or [],
				"warnings": sync_result.warnings,
			}
			logger.info(
				"rag.sync.complete status=ok jobs=%s companies=%s indexed_chunks=%s latency_ms=%s",
				sync_result.fetched_jobs,
				sync_result.fetched_companies,
				sync_result.indexed_chunks,
				latency_ms,
			)
			return {"status": "ok", "lastSyncAt": self._last_sync_at, "syncState": self._sync_status, **self._last_sync_result}

	def _schedule_background_sync(self) -> None:
		if self._sync_task and not self._sync_task.done():
			return

		async def _runner() -> None:
			try:
				await self.ensure_index_fresh(force=True)
			except Exception as exc:
				logger.exception("rag.sync.background.failed error=%s", exc)

		self._sync_task = asyncio.create_task(_runner())

	async def answer(
		self,
		query: str,
		conversation_history: Optional[List[Dict[str, str]]] = None,
		filters: Optional[Dict[str, Any]] = None,
		extra_context: Optional[str] = None,
		profile_name: Optional[str] = None,
	) -> Dict[str, Any]:
		started = time.perf_counter()
		self._metrics["chatRequests"] += 1
		collection_count = 0
		try:
			collection_count = self.vector_store.count()
		except Exception as exc:
			logger.warning("rag.pipeline.vector_store.count_failed error=%s", exc)

		if collection_count <= 0:
			empty_index_answer = (
				"Du lieu RAG hien chua duoc nap vao he thong. "
				"Hay thuc hien reload/sync du lieu tu backend truoc khi hoi dap."
			)
			formatted_answer = format_chat_response(empty_index_answer, [])
			structured_answer = build_structured_chat_response(empty_index_answer, [])
			total_latency_ms = round((time.perf_counter() - started) * 1000, 2)
			self._metrics["lastRetrievalLatencyMs"] = 0.0
			self._metrics["lastGenerationLatencyMs"] = 0.0
			self._metrics["lastPipelineLatencyMs"] = total_latency_ms
			self._metrics["lastRetrievedChunks"] = 0
			self._metrics["lastTopScore"] = None
			self.retrieval.metrics.empty_retrieval_count += 1
			self.retrieval.metrics.fallback_trigger_count += 1

			debug_mode = True if (str(os.getenv("DEBUG", "0")).lower() in {"1", "true", "yes"}) else False
			response = {
				"success": False,
				"version": "2.1.0",
				"conversationId": None,
				"message": {
					"id": str(uuid.uuid4()),
					"role": "assistant",
					"content": formatted_answer,
					"createdAt": datetime.utcnow().isoformat(),
				},
				"data": {"intent": "jobs_search", "jobs": [], "total": 0},
				"retrieval": {"profile": self.retrieval.last_profile.name, "topScore": None, "count": 0, "fallbackTriggered": True},
				"meta": {
					"latencyMs": total_latency_ms,
					"retrieval": {"latencyMs": 0.0, "rerankLatencyMs": None, "contextPackingLatencyMs": 0.0},
					"generation": {"model": QWEN_MODEL_NAME, "latencyMs": 0.0, "promptTokens": None, "completionTokens": None},
				},
			}
			if debug_mode:
				response["rag_debug"] = {"sync": {**self._last_sync_result, "status": self._sync_status, "lastError": self._last_sync_error}}
			return response

		retrieval_error: Optional[str] = None
		retrieval_timeout_seconds = max(0.1, RAG_RETRIEVAL_TIMEOUT_MS / 1000)
		try:
			retrieval = await asyncio.wait_for(
				asyncio.to_thread(self.retrieval.retrieve, query, filters, profile_name),
				timeout=retrieval_timeout_seconds,
			)
		except asyncio.TimeoutError:
			retrieval_error = f"retrieval-timeout-{RAG_RETRIEVAL_TIMEOUT_MS}ms"
			self.retrieval.metrics.fallback_trigger_count += 1
			logger.warning("rag.pipeline.retrieval.timeout timeout_ms=%s query=%r", RAG_RETRIEVAL_TIMEOUT_MS, query[:120])
			retrieval = RetrievalResult(
				query=query,
				items=[],
				latency_ms=round(retrieval_timeout_seconds * 1000, 2),
				cached=False,
			)
		except Exception as exc:
			retrieval_error = str(exc)
			self.retrieval.metrics.fallback_trigger_count += 1
			logger.exception("rag.pipeline.retrieval.failed query=%r error=%s", query[:120], exc)
			retrieval = RetrievalResult(query=query, items=[], latency_ms=0.0, cached=False)
		packing_started = time.perf_counter()
		packing = pack_context(
			retrieval.items,
			max_context_tokens=self.retrieval.last_context_budget_tokens,
			max_chunk_tokens=RAG_MAX_CHUNK_TOKENS,
		)
		packing_latency_ms = round((time.perf_counter() - packing_started) * 1000, 2)

		fallback = detect_low_confidence(
			retrieval.items, # type: ignore
			min_top_score=0.42,
			min_item_count=2,
		)
		if retrieval_error:
			fallback = detect_low_confidence([], min_top_score=1.0, min_item_count=999)
			self.retrieval.last_fallback = {
				"triggered": True,
				"stage": "retrieval",
				"reason": retrieval_error,
				"runtimeError": retrieval_error,
			}
		packed_items = [
			RetrievedChunk(
				chunk_id=item.chunk_id,
				text=item.text,
				metadata=item.metadata,
				distance=0.0,
				rerank_score=item.rerank_score,
				source_key=item.source_key,
			)
			for item in packing.items
		]
		prompt = build_chat_prompt(
			query=query,
			items=packed_items,
			history=conversation_history or [],
			extra_context=extra_context,
		)

		if fallback.low_confidence and fallback.safe_response:
			generation_answer = (
				"Truy xuat du lieu dang bi cham hoac bi timeout. "
				"Vui long thu lai sau, hoac thu query cu the hon nhu ky nang, dia diem, cong ty."
				if retrieval_error
				else fallback.safe_response
			)
			generation_latency_ms = 0.0
			self._metrics["lastGenerationLatencyMs"] = generation_latency_ms
			self.retrieval.metrics.hallucination_fallback_count += 1
		else:
			generation_started = time.perf_counter()
			generation = await asyncio.to_thread(self.generation.generate, prompt)
			generation_latency_ms = round((time.perf_counter() - generation_started) * 1000, 2)
			generation_answer = generation.answer
		total_latency_ms = round((time.perf_counter() - started) * 1000, 2)
		top_score = max((item.rerank_score for item in retrieval.items), default=None)

		logger.info(
			"rag.pipeline.complete retrieval_latency_ms=%s generation_latency_ms=%s total_latency_ms=%s chunks=%s top_score=%s last_sync_at=%s sync_state=%s",
			retrieval.latency_ms,
			generation_latency_ms,
			total_latency_ms,
			len(retrieval.items),
			top_score,
			self._last_sync_at,
			self._sync_status,
		)

		sources = [
			{
				"entityType": item.metadata.get("entity_type"),
				"sourceId": item.metadata.get("source_id"),
				"id": item.metadata.get("job_id") or item.metadata.get("company_id"),
				"title": item.metadata.get("title"),
				"company": item.metadata.get("company"),
				"location": item.metadata.get("location") or item.metadata.get("city"),
				"salary": item.metadata.get("salary"),
				"description": item.metadata.get("description"),
				"skills": item.metadata.get("skills"),
				"industry": item.metadata.get("industry"),
				"url": item.metadata.get("url"),
				"distance": item.distance,
				"score": item.rerank_score,
				"chunkId": item.chunk_id,
			}
			for item in retrieval.items
		]
		# Build production-friendly response structure
		formatted_answer = format_chat_response(generation_answer, sources)

		debug_mode = True if (str(os.getenv("DEBUG", "0")).lower() in {"1", "true", "yes"}) else False

		# Map retrieval items into frontend-friendly job items
		jobs_items = []
		for item in retrieval.items:
			md = item.metadata or {}
			jobs_items.append(
				{
					"id": str(md.get("job_id") or md.get("source_id") or md.get("id") or ""),
					"title": md.get("title"),
					"company": md.get("company"),
					"location": md.get("location") or md.get("city"),
					"salary": md.get("salary"),
					"description": md.get("description"),
					"skills": [s.strip() for s in str(md.get("skills") or "").split(",") if s.strip()],
					"jobType": md.get("job_type") or md.get("work_type"),
					"score": float(item.rerank_score) if item.rerank_score is not None else None,
					"url": md.get("url"),
				}
			)

		retrieval_summary = {
			"profile": self.retrieval.last_profile.name,
			"topScore": top_score,
			"count": len(retrieval.items),
			"fallbackTriggered": fallback.low_confidence,
		}

		meta = {
			"latencyMs": total_latency_ms,
			"retrieval": {
				"latencyMs": retrieval.latency_ms,
				"rerankLatencyMs": getattr(self.retrieval.metrics, "rerank_latency_ms", None),
				"contextPackingLatencyMs": packing_latency_ms,
			},
			"generation": {
				"model": QWEN_MODEL_NAME,
				"latencyMs": generation_latency_ms,
				"promptTokens": None if fallback.low_confidence else getattr(generation, "prompt_tokens_estimate", None),
				"completionTokens": None if fallback.low_confidence else getattr(generation, "completion_tokens_estimate", None),
			},
		}

		message_obj = {
			"id": str(uuid.uuid4()),
			"role": "assistant",
			"content": formatted_answer,
			"createdAt": datetime.utcnow().isoformat(),
		}

		response: Dict[str, Any] = {
			"success": True,
			"version": "2.1.0",
			"conversationId": (conversation_history[0]["conversationId"] if conversation_history and isinstance(conversation_history, list) and conversation_history and conversation_history[0].get("conversationId") else None),
			"message": message_obj,
			"data": {
				"intent": "jobs_search",
				"jobs": jobs_items,
				"total": len(retrieval.items),
			},
			"retrieval": retrieval_summary,
			"meta": meta,
		}

		# attach debug-only fields when debug mode enabled
		if debug_mode:
			response["rag_debug"] = {
				"sync": {"lastSyncAt": self._last_sync_at, "status": self._sync_status, "lastError": self._last_sync_error, **self._last_sync_result},
				"raw_retrieval_metrics": self.retrieval.metrics.to_dict(),
				"raw_items": [
					{"chunkId": it.chunk_id, "score": it.rerank_score, "metadata": it.metadata, "preview": it.text[:500]} for it in retrieval.items
				],
				"promptPreview": prompt[:2000],
			}

		# update internal metrics
		self._metrics["lastRetrievalLatencyMs"] = retrieval.latency_ms
		self._metrics["lastGenerationLatencyMs"] = generation_latency_ms
		self._metrics["lastPipelineLatencyMs"] = total_latency_ms
		self._metrics["lastRetrievedChunks"] = len(retrieval.items)
		self._metrics["lastTopScore"] = top_score
		self.retrieval.metrics.context_packing_latency_ms = packing_latency_ms

		return response

	def health(self) -> Dict[str, Any]:
		import torch  # type: ignore

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
			"syncStatus": self._sync_status,
			"lastSyncError": self._last_sync_error,
		}

	async def retrieval_debug(
		self,
		query: str,
		filters: Optional[Dict[str, Any]] = None,
		profile_name: Optional[str] = None,
	) -> Dict[str, Any]:
		retrieval = self.retrieval.retrieve(query, filters=filters, profile_name=profile_name)
		return {
			"query": query,
			"count": len(retrieval.items),
			"latencyMs": retrieval.latency_ms,
			"cached": retrieval.cached,
			"profile": self.retrieval.last_profile.name,
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

	def retrieval_profiles(self) -> Dict[str, Any]:
		return {
			"defaultProfile": RAG_DEFAULT_PROFILE,
			"currentProfile": self.retrieval.last_profile.name if self.retrieval.last_profile else None,
			"profiles": {
				name: {
					"candidateK": profile.candidate_k,
					"topK": profile.top_k,
					"rerankWithCrossEncoder": profile.rerank_with_cross_encoder,
					"maxContextTokens": profile.max_context_tokens,
					"normalization": profile.normalization,
					"maxChunksPerSource": profile.max_chunks_per_source,
					"weights": {
						"semantic": profile.semantic_weight,
						"bm25": profile.bm25_weight,
						"title": profile.title_weight,
						"company": profile.company_weight,
						"description": profile.description_weight,
						"skills": profile.skills_weight,
						"category": profile.category_weight,
						"entityBias": profile.entity_bias_weight,
					},
				}
				for name, profile in PROFILES.items()
			},
		}

	async def benchmark_retrieval(self, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
		evaluation_cases = [
			EvaluationCase(
				query=str(case.get("query") or ""),
				expected_terms=[str(term) for term in case.get("expectedTerms", [])],
				profile=str(case.get("profile") or "balanced"),
			) # type: ignore
			for case in cases
			if str(case.get("query") or "").strip()
		]
		return await asyncio.to_thread(benchmark_retrieval, self.retrieval, evaluation_cases) # type: ignore

	async def list_indexed_jobs(self, page: int = 1, limit: int = 20) -> Dict[str, Any]:
		page = max(1, page)
		limit = max(1, min(limit, 100))
		raw = self.vector_store.get(limit=min(limit * 6, 600), offset=max(0, (page - 1) * limit * 2))
		metadatas = raw.get("metadatas") or []
		documents = raw.get("documents") or []

		deduped: List[Dict[str, Any]] = []
		seen_job_ids: set[str] = set()

		for idx, metadata in enumerate(metadatas):
			if not metadata or metadata.get("entity_type") != "job":
				continue

			doc_text = str(documents[idx] if idx < len(documents) else "")
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
					"description": _public_description(metadata.get("description"), doc_text, "Job Description:", ["Requirements:", "Benefits:"]),
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

	async def list_indexed_companies(self, page: int = 1, limit: int = 20) -> Dict[str, Any]:
		page = max(1, page)
		limit = max(1, min(limit, 100))
		raw = self.vector_store.get(
			limit=min(limit * 6, 600),
			offset=max(0, (page - 1) * limit * 2),
			where={"entity_type": "company"},
		)
		metadatas = raw.get("metadatas") or []
		documents = raw.get("documents") or []

		deduped: List[Dict[str, Any]] = []
		seen_company_ids: set[str] = set()
		for idx, metadata in enumerate(metadatas):
			if not metadata:
				continue
			company_id = str(metadata.get("company_id") or metadata.get("source_id") or "")
			if not company_id or company_id in seen_company_ids:
				continue

			doc_text = str(documents[idx] if idx < len(documents) else "")
			seen_company_ids.add(company_id)
			deduped.append(
				{
					"companyId": company_id,
					"company": metadata.get("company"),
					"industry": metadata.get("industry"),
					"companyType": metadata.get("company_type"),
					"size": metadata.get("size"),
					"location": metadata.get("location"),
					"description": _public_description(metadata.get("description"), doc_text, "Description:", ["Why Join:"]),
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

	async def get_indexed_company_detail(self, company_id: str) -> Dict[str, Any]:
		raw = self.vector_store.get(limit=20, offset=0, where={"company_id": str(company_id)})
		metadatas = raw.get("metadatas") or []
		documents = raw.get("documents") or []
		if not metadatas:
			raise ValueError(f"Company {company_id} not found in Chroma index")

		first = metadatas[0]
		contexts = [doc[:600] for doc in documents if isinstance(doc, str)]
		return {
			"companyId": str(first.get("company_id") or company_id),
			"company": first.get("company"),
			"industry": first.get("industry"),
			"companyType": first.get("company_type"),
			"size": first.get("size"),
			"location": first.get("location"),
			"description": _public_description(first.get("description"), documents[0] if documents else "", "Description:", ["Why Join:"]),
			"url": first.get("url"),
			"updatedAt": first.get("updated_at"),
			"chunks": len(metadatas),
			"contexts": contexts,
		}

	def metrics(self) -> Dict[str, Any]:
		return {
			"sync": {
				"status": self._sync_status,
				"lastSyncAt": self._last_sync_at,
				"lastSyncError": self._last_sync_error,
				**self._last_sync_result,
			},
			"pipeline": dict(self._metrics),
			"retrieval": self.retrieval.metrics.to_dict(),
			"vectorStore": self.vector_store.stats(),
			"generation": {
				"device": self.generation.device,
				"dtype": str(self.generation.dtype),
				"model": QWEN_MODEL_NAME,
			},
			"embedding": {
				"device": self.embedding.device,
				"batchSize": self.embedding.batch_size,
			},
		}

	def prometheus_metrics(self) -> str:
		sync_last = self._last_sync_at or 0
		metrics = {
			"rag_sync_status": 1 if self._sync_status == "idle" else 0,
			"rag_sync_last_timestamp_seconds": sync_last,
			"rag_chat_requests_total": self._metrics.get("chatRequests", 0) or 0,
			"rag_sync_requests_total": self._metrics.get("syncRequests", 0) or 0,
			"rag_retrieval_latency_ms": self._metrics.get("lastRetrievalLatencyMs") or 0,
			"rag_generation_latency_ms": self._metrics.get("lastGenerationLatencyMs") or 0,
			"rag_pipeline_latency_ms": self._metrics.get("lastPipelineLatencyMs") or 0,
			"rag_retrieved_chunks": self._metrics.get("lastRetrievedChunks", 0) or 0,
			"rag_retrieval_top_score": self._metrics.get("lastTopScore") or 0,
			"rag_vectorstore_documents": self.vector_store.count(),
			"rag_empty_retrieval_total": self.retrieval.metrics.empty_retrieval_count,
			"rag_duplicate_removed_total": self.retrieval.metrics.duplicate_removed_count,
			"rag_fallback_trigger_total": self.retrieval.metrics.fallback_trigger_count,
			"rag_hallucination_fallback_total": self.retrieval.metrics.hallucination_fallback_count,
			"rag_rerank_latency_ms": self.retrieval.metrics.rerank_latency_ms,
			"rag_context_packing_latency_ms": self.retrieval.metrics.context_packing_latency_ms,
		}
		lines = [
			"# HELP rag_sync_status 1 when sync is idle, 0 otherwise",
			"# TYPE rag_sync_status gauge",
		]
		for key, value in metrics.items():
			metric_type = "gauge"
			if key.endswith("_total"):
				metric_type = "counter"
			lines.append(f"# TYPE {key} {metric_type}")
			lines.append(f"{key} {value}")
		return "\n".join(lines) + "\n"


_pipeline_singleton: Optional[RAGPipelineService] = None


def get_rag_pipeline() -> RAGPipelineService:
	global _pipeline_singleton
	if _pipeline_singleton is None:
		_pipeline_singleton = RAGPipelineService()
	return _pipeline_singleton
