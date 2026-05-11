"""Ingestion pipeline: fetch -> normalize -> chunk -> embed -> replace index."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Sequence, Tuple

from app.services.backend_api_client import get_backend_client

from .embedding import EmbeddingService
from .schemas import ChunkRecord, CompanyRecord, JobRecord, SyncResult
from .settings import (
    EMBED_BATCH_SIZE,
    RAG_BACKEND_MAX_PAGES,
    RAG_BACKEND_PAGE_SIZE,
    RAG_CHUNK_OVERLAP_CHARS,
    RAG_CHUNK_SIZE_CHARS,
    RAG_COMPANY_MAX_PAGES,
)
from .vector_store import RAGVectorStore

logger = logging.getLogger(__name__)


class RAGIngestionService:
    """Keeps the vector index in sync with backend jobs and companies APIs."""

    def __init__(self, vector_store: RAGVectorStore, embedding: EmbeddingService) -> None:
        self.backend = get_backend_client()
        self.vector_store = vector_store
        self.embedding = embedding

    async def sync_from_backend(self) -> SyncResult:
        start = time.perf_counter()
        errors: List[str] = []
        warnings: List[str] = []

        jobs, job_pages, job_errors = await self._fetch_jobs()
        companies, company_pages, company_errors = await self._fetch_companies()
        errors.extend(job_errors)
        errors.extend(company_errors)

        if not jobs:
            warnings.append("No jobs fetched from backend")

        job_chunks = self._chunk_job_records(jobs)
        company_chunks = self._chunk_company_records(companies)
        all_chunks = self._dedupe_chunks(job_chunks + company_chunks)

        stale_removed = self.vector_store.count() if all_chunks else 0
        if all_chunks:
            previous_count = self.vector_store.reset_collection()
            stale_removed = previous_count
            self._index_chunks_in_batches(all_chunks)
        else:
            logger.warning("Skipping vector store reset because no chunks were produced")

        took = time.perf_counter() - start
        logger.info(
            "RAG sync done jobs=%s companies=%s chunks=%s took=%.2fs",
            len(jobs),
            len(companies),
            len(all_chunks),
            took,
        )
        return SyncResult(
            fetched_jobs=len(jobs),
            fetched_companies=len(companies),
            indexed_chunks=len(all_chunks),
            indexed_job_chunks=len(job_chunks),
            indexed_company_chunks=len(company_chunks),
            skipped_jobs=max(0, len(job_chunks) + len(company_chunks) - len(all_chunks)),
            stale_chunks_removed=stale_removed,
            backend_pages=job_pages + company_pages,
            took_seconds=round(took, 3),
            errors=errors or None,
            warnings=warnings,
        )

    async def _fetch_jobs(self) -> Tuple[List[JobRecord], int, List[str]]:
        seen_ids: set[str] = set()
        records: List[JobRecord] = []
        errors: List[str] = []
        pages = 0

        for page in range(1, RAG_BACKEND_MAX_PAGES + 1):
            payload = await self.backend.get_all_jobs(page=page, limit=RAG_BACKEND_PAGE_SIZE, active_only=True)
            if "error" in payload:
                errors.append(f"jobs page {page}: {payload['error']}")
                break

            jobs_raw = payload.get("jobs") or payload.get("data") or payload.get("items") or []
            if not jobs_raw:
                break

            pages = page
            for raw in jobs_raw:
                normalized = self._normalize_job(raw)
                if not normalized.job_id or normalized.job_id in seen_ids:
                    continue
                seen_ids.add(normalized.job_id)
                records.append(normalized)

            total = payload.get("total")
            limit = payload.get("limit") or RAG_BACKEND_PAGE_SIZE
            if isinstance(total, int) and total > 0 and len(records) >= total:
                break
            if len(jobs_raw) < limit:
                break

        return records, pages, errors

    async def _fetch_companies(self) -> Tuple[List[CompanyRecord], int, List[str]]:
        seen_ids: set[str] = set()
        records: List[CompanyRecord] = []
        errors: List[str] = []
        pages = 0

        for page in range(1, RAG_COMPANY_MAX_PAGES + 1):
            payload = await self.backend.get_all_companies(page=page)
            if "error" in payload:
                errors.append(f"companies page {page}: {payload['error']}")
                break

            companies_raw = payload.get("companies") or payload.get("data") or payload.get("items") or []
            if not companies_raw:
                break

            pages = page
            for raw in companies_raw:
                normalized = self._normalize_company(raw)
                if not normalized.company_id or normalized.company_id in seen_ids:
                    continue
                seen_ids.add(normalized.company_id)
                records.append(normalized)

            total = payload.get("total")
            if isinstance(total, int) and total > 0 and len(records) >= total:
                break
            if len(companies_raw) < 10:
                break

        return records, pages, errors

    def _index_chunks_in_batches(self, chunks: Sequence[ChunkRecord]) -> int:
        indexed = 0
        for i in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[i : i + EMBED_BATCH_SIZE]
            vectors = self.embedding.encode([c.text for c in batch])
            indexed += self.vector_store.upsert_chunks(batch, vectors)
        return indexed

    def _normalize_job(self, raw: Dict) -> JobRecord:
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        company_obj = raw.get("company") or raw.get("Company") or {}
        category_obj = raw.get("category") or raw.get("Category") or {}
        job_type_obj = raw.get("jobType") or raw.get("JobType") or {}

        job_id = str(raw.get("id") or raw.get("job_id") or raw.get("jobId") or raw.get("job_post_id") or "")
        company = str(
            raw.get("company_name")
            or raw.get("company")
            or company_obj.get("company_name")
            or company_obj.get("name")
            or raw.get("name")
            or ""
        )
        location = str(
            raw.get("location")
            or raw.get("workLocation")
            or raw.get("work_location")
            or company_obj.get("city")
            or ""
        )
        category = str(raw.get("category_name") or raw.get("category") or category_obj.get("name") or "")
        job_type = str(raw.get("job_type") or raw.get("job_type_name") or job_type_obj.get("job_type") or "")

        skills_raw = raw.get("skills") or raw.get("skills_text") or []
        return JobRecord(
            job_id=job_id,
            title=str(raw.get("title") or raw.get("job_title") or raw.get("name") or ""),
            company=company,
            location=location,
            salary=str(raw.get("salary") or raw.get("salary_range") or raw.get("salaryRange") or ""),
            skills=self._normalize_skills(skills_raw),
            category=category,
            description=str(raw.get("description") or raw.get("job_description") or ""),
            requirements=self._normalize_requirements(raw.get("requirements") or raw.get("candidate_requirements") or ""),
            benefits=str(raw.get("benefits") or ""),
            work_type=str(raw.get("work_type") or raw.get("workType") or ""),
            job_type=job_type,
            level=str(raw.get("level") or ""),
            url=self._build_job_url(job_id, raw),
            updated_at=str(raw.get("updated_at") or raw.get("updatedAt") or raw.get("updatedDate") or now_iso),
        )

    def _normalize_company(self, raw: Dict) -> CompanyRecord:
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        company_id = str(raw.get("company_id") or raw.get("id") or "")
        location = ", ".join([str(v).strip() for v in [raw.get("city"), raw.get("country")] if str(v or "").strip()])
        return CompanyRecord(
            company_id=company_id,
            name=str(raw.get("company_name") or raw.get("name") or ""),
            industry=str(raw.get("company_industry") or raw.get("industry") or ""),
            company_type=str(raw.get("company_type") or ""),
            size=str(raw.get("company_size") or raw.get("size") or ""),
            location=location,
            website=str(raw.get("company_website_url") or raw.get("website") or ""),
            email=str(raw.get("company_email") or raw.get("email") or ""),
            description=str(raw.get("profile_description") or raw.get("description") or ""),
            key_skills=self._normalize_skills(raw.get("key_skills") or ""),
            why_join=str(raw.get("why_love_working_here") or ""),
            updated_at=str(raw.get("updated_date") or raw.get("updatedAt") or raw.get("created_date") or now_iso),
        )

    @staticmethod
    def _normalize_skills(skills_raw: object) -> str:
        if isinstance(skills_raw, list):
            return ", ".join([str(s).strip() for s in skills_raw if str(s).strip()])
        return str(skills_raw or "")

    @staticmethod
    def _normalize_requirements(requirements_raw: object) -> str:
        if isinstance(requirements_raw, list):
            return "\n".join([str(item).strip() for item in requirements_raw if str(item).strip()])
        return str(requirements_raw or "")

    @staticmethod
    def _build_job_url(job_id: str, raw: Dict) -> str:
        if raw.get("url") or raw.get("job_url"):
            return str(raw.get("url") or raw.get("job_url"))
        return f"/jobs/{job_id}" if job_id else ""

    def _chunk_job_records(self, records: Sequence[JobRecord]) -> List[ChunkRecord]:
        chunks: List[ChunkRecord] = []
        for record in records:
            text = self._build_job_document_text(record)
            for idx, piece in enumerate(self._chunk_text(text)):
                chunks.append(
                    ChunkRecord(
                        chunk_id=f"job-{record.job_id}-c{idx}",
                        text=piece,
                        metadata={
                            "entity_type": "job",
                            "source_id": record.job_id,
                            "source_key": f"job:{record.job_id}",
                            "job_id": record.job_id,
                            "title": record.title,
                            "company": record.company,
                            "location": record.location,
                            "salary": record.salary,
                            "skills": record.skills,
                            "category": record.category,
                            "work_type": record.work_type,
                            "job_type": record.job_type,
                            "level": record.level,
                            "url": record.url,
                            "updated_at": record.updated_at,
                            "chunk_index": idx,
                        },
                    )
                )
        return chunks

    def _chunk_company_records(self, records: Sequence[CompanyRecord]) -> List[ChunkRecord]:
        chunks: List[ChunkRecord] = []
        for record in records:
            text = self._build_company_document_text(record)
            for idx, piece in enumerate(self._chunk_text(text)):
                chunks.append(
                    ChunkRecord(
                        chunk_id=f"company-{record.company_id}-c{idx}",
                        text=piece,
                        metadata={
                            "entity_type": "company",
                            "source_id": record.company_id,
                            "source_key": f"company:{record.company_id}",
                            "company_id": record.company_id,
                            "company": record.name,
                            "title": record.name,
                            "industry": record.industry,
                            "company_type": record.company_type,
                            "size": record.size,
                            "location": record.location,
                            "skills": record.key_skills,
                            "url": record.website,
                            "updated_at": record.updated_at,
                            "chunk_index": idx,
                        },
                    )
                )
        return chunks

    @staticmethod
    def _build_job_document_text(record: JobRecord) -> str:
        return (
            f"Job ID: {record.job_id}\n"
            f"Title: {record.title}\n"
            f"Company: {record.company}\n"
            f"Location: {record.location}\n"
            f"Salary: {record.salary}\n"
            f"Category: {record.category}\n"
            f"Work Type: {record.work_type}\n"
            f"Job Type: {record.job_type}\n"
            f"Level: {record.level}\n"
            f"Skills: {record.skills}\n"
            f"Description: {record.description}\n"
            f"Requirements: {record.requirements}\n"
            f"Benefits: {record.benefits}\n"
            f"URL: {record.url}\n"
        )

    @staticmethod
    def _build_company_document_text(record: CompanyRecord) -> str:
        return (
            f"Company ID: {record.company_id}\n"
            f"Company: {record.name}\n"
            f"Industry: {record.industry}\n"
            f"Company Type: {record.company_type}\n"
            f"Company Size: {record.size}\n"
            f"Location: {record.location}\n"
            f"Website: {record.website}\n"
            f"Email: {record.email}\n"
            f"Key Skills: {record.key_skills}\n"
            f"Description: {record.description}\n"
            f"Why join: {record.why_join}\n"
        )

    @staticmethod
    def _chunk_text(text: str) -> List[str]:
        if len(text) <= RAG_CHUNK_SIZE_CHARS:
            return [text]

        chunks: List[str] = []
        start = 0
        step = max(1, RAG_CHUNK_SIZE_CHARS - RAG_CHUNK_OVERLAP_CHARS)
        while start < len(text):
            end = min(len(text), start + RAG_CHUNK_SIZE_CHARS)
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start += step
        return chunks

    @staticmethod
    def _dedupe_chunks(chunks: Sequence[ChunkRecord]) -> List[ChunkRecord]:
        deduped: List[ChunkRecord] = []
        seen: set[tuple[str, str]] = set()
        for chunk in chunks:
            key = (
                str(chunk.metadata.get("source_key") or chunk.chunk_id),
                " ".join(chunk.text.split()),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(chunk)
        return deduped
