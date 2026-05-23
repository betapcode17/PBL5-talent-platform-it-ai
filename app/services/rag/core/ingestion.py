"""Production-ready ingestion service for RAG indexing.

This version keeps the backend payload shape as the source of truth.
It intentionally avoids inventing `skills` fields when the backend API does
not provide them.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from html import unescape
from typing import Any, Dict, List, Optional

from app.services.backend_api_client import get_backend_client

from ..infra.embedding import EmbeddingService
from ..schemas_and_settings.schemas import ChunkRecord, SyncResult
from ..schemas_and_settings.settings import (
    RAG_BACKEND_MAX_PAGES,
    RAG_BACKEND_PAGE_SIZE,
    RAG_CHUNK_MAX_TOKENS,
    RAG_COMPANY_MAX_PAGES,
)
from ..stores.fulltext_store import RAGFullTextStore
from ..stores.vector_store import RAGVectorStore

logger = logging.getLogger(__name__)

MAX_CONCURRENT_TASKS = 10
VECTOR_BATCH_SIZE = 64
CHUNK_OVERLAP = 40


class IngestionService:
    """Production-ready ingestion pipeline."""

    def __init__(
        self,
        vector_store: RAGVectorStore,
        embedding: EmbeddingService,
        fulltext_store: Optional[RAGFullTextStore] = None,
    ) -> None:
        self.vector_store = vector_store
        self.embedding = embedding
        self.fulltext_store = fulltext_store
        self.backend_client = get_backend_client()

    @staticmethod
    def _safe_str(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @staticmethod
    def _clean_html(text: str) -> str:
        if not text:
            return ""
        text = unescape(text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _content_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        cleaned: Dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                cleaned[key] = ""
                continue
            if isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
                continue
            if isinstance(value, list):
                cleaned[key] = [str(item).strip() for item in value if item is not None]
                continue
            cleaned[key] = str(value)
        return cleaned

    @staticmethod
    def _split_text(text: str, max_words: int = RAG_CHUNK_MAX_TOKENS, overlap: int = CHUNK_OVERLAP) -> List[str]:
        text = text.strip()
        if not text:
            return []

        words = text.split()
        if len(words) <= max_words:
            return [text]

        chunks: List[str] = []
        start = 0
        while start < len(words):
            end = start + max_words
            chunks.append(" ".join(words[start:end]))
            if end >= len(words):
                break
            start = end - overlap
        return chunks

    def chunk_job_records(self, job_payload: Dict[str, Any]) -> List[ChunkRecord]:
        job_id = self._safe_str(job_payload.get("id") or job_payload.get("job_id"))
        title = self._safe_str(job_payload.get("title") or job_payload.get("position"))
        company = self._safe_str(job_payload.get("company") or job_payload.get("employer"))
        description = self._clean_html(self._safe_str(job_payload.get("description") or job_payload.get("summary")))
        location = self._safe_str(job_payload.get("location") or job_payload.get("city"))
        category = self._safe_str(job_payload.get("category") or job_payload.get("category_name"))
        salary = self._safe_str(job_payload.get("salary") or job_payload.get("salary_range"))
        job_type = self._safe_str(job_payload.get("job_type") or job_payload.get("work_type"))
        updated_at = self._safe_str(job_payload.get("updated_at") or job_payload.get("updatedDate"))

        base_text = description or f"{title} at {company}"
        if not base_text.strip():
            logger.warning("Skipping empty job job_id=%s", job_id)
            return []

        content_hash = self._content_hash(base_text)
        chunks_text = self._split_text(base_text)
        results: List[ChunkRecord] = []

        for idx, chunk_text in enumerate(chunks_text):
            enriched_text = f"""
Job Title: {title}

Company: {company}

Location: {location}

Category: {category}

Salary: {salary}

Job Type: {job_type}

Content:
{chunk_text}
""".strip()

            metadata = self._clean_metadata(
                {
                    "entity_type": "job",
                    "job_id": job_id,
                    "title": title,
                    "company": company,
                    "location": location,
                    "category": category,
                    "salary": salary,
                    "job_type": job_type,
                    "updated_at": updated_at,
                    "content_hash": content_hash,
                }
            )

            results.append(
                ChunkRecord(
                    chunk_id=f"job:{job_id}:chunk:{idx}",
                    text=enriched_text,
                    metadata=metadata,
                )
            )

        return results

    def chunk_company_records(self, company_payload: Dict[str, Any]) -> List[ChunkRecord]:
        company_id = self._safe_str(company_payload.get("id") or company_payload.get("company_id"))
        company_name = self._safe_str(company_payload.get("name") or company_payload.get("company"))
        description = self._clean_html(self._safe_str(company_payload.get("description") or company_payload.get("summary")))
        industry = self._safe_str(company_payload.get("industry"))
        location = self._safe_str(company_payload.get("location") or company_payload.get("city"))
        website = self._safe_str(company_payload.get("website") or company_payload.get("url"))
        updated_at = self._safe_str(company_payload.get("updated_at") or company_payload.get("updatedDate"))

        text = description or company_name
        if not text.strip():
            logger.warning("Skipping empty company company_id=%s", company_id)
            return []

        metadata = self._clean_metadata(
            {
                "entity_type": "company",
                "company_id": company_id,
                "company": company_name,
                "industry": industry,
                "location": location,
                "website": website,
                "updated_at": updated_at,
            }
        )

        return [
            ChunkRecord(
                chunk_id=f"company:{company_id}:chunk:0",
                text=text,
                metadata=metadata,
            )
        ]

    def _upsert_chunks(self, chunks: List[ChunkRecord]) -> None:
        if not chunks:
            return

        for i in range(0, len(chunks), VECTOR_BATCH_SIZE):
            batch = chunks[i : i + VECTOR_BATCH_SIZE]
            self.vector_store.upsert_chunks(batch, embeddings_provider=self.embedding)
            if self.fulltext_store:
                self.fulltext_store.upsert_chunks(batch)

    @staticmethod
    def _extract_items(payload: Any) -> List[Dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]

        if not isinstance(payload, dict):
            return []

        for key in ("items", "data", "results", "rows", "jobs", "companies"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, dict):
                for nested_key in ("items", "data", "results", "rows", "jobs", "companies"):
                    nested = value.get(nested_key)
                    if isinstance(nested, list):
                        return [item for item in nested if isinstance(item, dict)]

        return []

    async def sync_from_backend(self) -> SyncResult:
        started = time.perf_counter()

        fetched_jobs = 0
        fetched_companies = 0
        indexed_chunks = 0
        indexed_job_chunks = 0
        indexed_company_chunks = 0
        backend_pages = 0
        errors: List[str] = []
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

        async def process_job(raw_job: Dict[str, Any]) -> int:
            async with semaphore:
                try:
                    normalized = self._normalize_job_payload(raw_job)
                    chunks = self.chunk_job_records(normalized)
                    await asyncio.to_thread(self._upsert_chunks, chunks)
                    return len(chunks)
                except Exception as exc:
                    logger.exception("Failed ingest job_id=%s", raw_job.get("id"))
                    errors.append(f"job:{raw_job.get('id')} -> {exc}")
                    return 0

        async def process_company(raw_company: Dict[str, Any]) -> int:
            async with semaphore:
                try:
                    normalized = self._normalize_company_payload(raw_company)
                    chunks = self.chunk_company_records(normalized)
                    await asyncio.to_thread(self._upsert_chunks, chunks)
                    return len(chunks)
                except Exception as exc:
                    logger.exception("Failed ingest company_id=%s", raw_company.get("id"))
                    errors.append(f"company:{raw_company.get('id')} -> {exc}")
                    return 0

        for page in range(1, RAG_BACKEND_MAX_PAGES + 1):
            backend_pages += 1
            jobs_payload = await self.backend_client.get_all_jobs(page=page, limit=RAG_BACKEND_PAGE_SIZE, active_only=True)
            job_items = self._extract_items(jobs_payload)
            if not job_items:
                break

            fetched_jobs += len(job_items)
            results = await asyncio.gather(*(process_job(job) for job in job_items))
            indexed_job_chunks += sum(results)
            indexed_chunks += sum(results)

            if len(job_items) < RAG_BACKEND_PAGE_SIZE:
                break

        for page in range(1, RAG_COMPANY_MAX_PAGES + 1):
            backend_pages += 1
            companies_payload = await self.backend_client.get_all_companies(page=page)
            company_items = self._extract_items(companies_payload)
            if not company_items:
                break

            fetched_companies += len(company_items)
            results = await asyncio.gather(*(process_company(company) for company in company_items))
            indexed_company_chunks += sum(results)
            indexed_chunks += sum(results)

            if len(company_items) < RAG_BACKEND_PAGE_SIZE:
                break

        took_seconds = round(time.perf_counter() - started, 3)
        return SyncResult(
            fetched_jobs=fetched_jobs,
            fetched_companies=fetched_companies,
            indexed_chunks=indexed_chunks,
            indexed_job_chunks=indexed_job_chunks,
            indexed_company_chunks=indexed_company_chunks,
            skipped_jobs=0,
            backend_pages=backend_pages,
            stale_chunks_removed=0,
            took_seconds=took_seconds,
            errors=errors or None,
            warnings=[],
        )

    @staticmethod
    def _normalize_job_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": payload.get("id") or payload.get("job_id"),
            "title": payload.get("title") or payload.get("name") or payload.get("position") or "",
            "company": payload.get("company") or payload.get("companyName") or payload.get("employer") or "",
            "description": payload.get("description") or payload.get("summary") or "",
            "location": payload.get("location") or payload.get("city") or "",
            "salary": payload.get("salary") or payload.get("salary_range") or "",
            "category": payload.get("category") or payload.get("category_name") or "",
            "job_type": payload.get("job_type") or payload.get("work_type") or "",
            "updated_at": payload.get("updated_at") or payload.get("updatedDate") or "",
        }

    @staticmethod
    def _normalize_company_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": payload.get("id") or payload.get("company_id"),
            "name": payload.get("name") or payload.get("company") or "",
            "description": payload.get("description") or payload.get("summary") or "",
            "industry": payload.get("industry") or "",
            "location": payload.get("location") or payload.get("city") or "",
            "website": payload.get("website") or payload.get("url") or "",
            "updated_at": payload.get("updated_at") or payload.get("updatedDate") or "",
        }
