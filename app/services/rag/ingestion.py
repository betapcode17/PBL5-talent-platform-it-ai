"""Ingestion pipeline: fetch -> normalize -> chunk -> embed -> replace index."""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from app.services.backend_api_client import get_backend_client

from .embedding import EmbeddingService
from .fulltext_store import RAGFullTextStore
from .schemas import ChunkRecord, CompanyRecord, JobRecord, SyncResult
from .settings import (
    EMBED_BATCH_SIZE,
    RAG_BACKEND_MAX_PAGES,
    RAG_BACKEND_PAGE_SIZE,
    RAG_CHUNK_OVERLAP_CHARS,
    RAG_CHUNK_SIZE_CHARS,
    RAG_COMPANY_MAX_PAGES,
    RAG_USE_FULLTEXT_SEARCH,
)
from .vector_store import RAGVectorStore

logger = logging.getLogger(__name__)


class RAGIngestionService:
    """Keeps the vector index in sync with backend jobs and companies APIs."""

    def __init__(self, vector_store: RAGVectorStore, embedding: EmbeddingService) -> None:
        self.backend = get_backend_client()
        self.vector_store = vector_store
        self.embedding = embedding
        self.fulltext_store = RAGFullTextStore() if RAG_USE_FULLTEXT_SEARCH else None

    async def sync_from_backend(self) -> SyncResult:
        start = time.perf_counter()
        errors: List[str] = []
        warnings: List[str] = []
        logger.info("rag.ingest.begin")

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
            if self.fulltext_store is not None:
                self.fulltext_store.rebuild_index(all_chunks)
        else:
            logger.warning("Skipping vector store reset because no chunks were produced")

        took = time.perf_counter() - start
        logger.info(
            "rag.ingest.complete jobs=%s companies=%s chunks=%s took_seconds=%.2f",
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
        company = self._extract_company_name(raw, company_obj)
        company_id = self._first_text(
            raw.get("company_id"),
            raw.get("companyId"),
            company_obj.get("company_id") if isinstance(company_obj, dict) else None,
            company_obj.get("companyId") if isinstance(company_obj, dict) else None,
            company_obj.get("id") if isinstance(company_obj, dict) else None,
        )
        city = self._first_text(raw.get("city"), company_obj.get("city") if isinstance(company_obj, dict) else None)
        location = self._extract_location(raw, company_obj)
        category_id = self._first_text(
            raw.get("category_id"),
            raw.get("categoryId"),
            category_obj.get("category_id") if isinstance(category_obj, dict) else None,
            category_obj.get("categoryId") if isinstance(category_obj, dict) else None,
            category_obj.get("id") if isinstance(category_obj, dict) else None,
        )
        category = self._first_text(raw.get("category_name"), category_obj.get("name"), raw.get("category"))
        job_type_id = self._first_text(
            raw.get("job_type_id"),
            raw.get("jobTypeId"),
            job_type_obj.get("job_type_id") if isinstance(job_type_obj, dict) else None,
            job_type_obj.get("jobTypeId") if isinstance(job_type_obj, dict) else None,
            job_type_obj.get("id") if isinstance(job_type_obj, dict) else None,
        )
        job_type = self._first_text(raw.get("job_type"), raw.get("job_type_name"), job_type_obj.get("job_type"))
        salary_min, salary_max = self._extract_salary_bounds(
            raw.get("salaryRange") or raw.get("salary_range"),
            raw.get("salary") or raw.get("salary_range") or raw.get("salaryRange"),
        )
        is_active = self._to_bool(raw.get("isActive", raw.get("is_active", True)))

        skills_raw = raw.get("skills") or raw.get("skills_text") or []
        return JobRecord(
            job_id=job_id,
            title=self._first_text(raw.get("title"), raw.get("job_title"), raw.get("name")),
            company_id=company_id,
            company=company,
            city=city,
            location=location,
            salary_min=salary_min,
            salary_max=salary_max,
            salary=self._normalize_salary_value(raw.get("salaryRange") or raw.get("salary_range") or raw.get("salary")),
            is_active=is_active,
            created_at=self._first_text(raw.get("created_at"), raw.get("createdAt"), raw.get("createdDate"), now_iso),
            category_id=category_id,
            skills=self._normalize_skills(skills_raw),
            category=category,
            job_type_id=job_type_id,
            description=self._first_text(raw.get("description"), raw.get("job_description")),
            requirements=self._normalize_requirements(raw.get("requirements") or raw.get("candidate_requirements") or ""),
            benefits=self._first_text(raw.get("benefits")),
            work_type=self._first_text(raw.get("work_type"), raw.get("workType")),
            job_type=job_type,
            level=self._first_text(raw.get("level")),
            url=self._build_job_url(job_id, raw),
            updated_at=self._first_text(raw.get("updated_at"), raw.get("updatedAt"), raw.get("updatedDate"), now_iso),
        )

    def _normalize_company(self, raw: Dict) -> CompanyRecord:
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        company_id = str(raw.get("company_id") or raw.get("id") or "")
        location = ", ".join([str(v).strip() for v in [raw.get("city"), raw.get("country")] if str(v or "").strip()])
        return CompanyRecord(
            company_id=company_id,
            name=self._first_text(raw.get("company_name"), raw.get("name")),
            industry=self._first_text(raw.get("company_industry"), raw.get("industry")),
            company_type=self._first_text(raw.get("company_type")),
            size=self._first_text(raw.get("company_size"), raw.get("size")),
            location=location,
            website=self._first_text(raw.get("company_website_url"), raw.get("website")),
            email=self._first_text(raw.get("company_email"), raw.get("email")),
            description=self._first_text(raw.get("profile_description"), raw.get("description")),
            key_skills=self._normalize_skills(raw.get("key_skills") or ""),
            why_join=self._first_text(raw.get("why_love_working_here")),
            updated_at=self._first_text(raw.get("updated_date"), raw.get("updatedAt"), raw.get("created_date"), now_iso),
        )

    @staticmethod
    def _normalize_skills(skills_raw: object) -> str:
        if isinstance(skills_raw, dict):
            return ", ".join(
                str(value).strip()
                for value in skills_raw.values()
                if str(value or "").strip() and str(value).strip().lower() not in {"none", "null"}
            )
        if isinstance(skills_raw, list):
            return ", ".join([str(s).strip() for s in skills_raw if str(s).strip()])
        return str(skills_raw or "").strip()

    @staticmethod
    def _normalize_requirements(requirements_raw: object) -> str:
        if isinstance(requirements_raw, dict):
            parts = [str(value).strip() for value in requirements_raw.values() if str(value or "").strip()]
            return "\n".join(parts)
        if isinstance(requirements_raw, list):
            return "\n".join([str(item).strip() for item in requirements_raw if str(item).strip()])
        return str(requirements_raw or "").strip()

    @staticmethod
    def _first_text(*values: object) -> str:
        for value in values:
            if value is None:
                continue
            if isinstance(value, dict):
                continue
            text = str(value).strip()
            if text and text.lower() not in {"none", "null"}:
                return text
        return ""

    @classmethod
    def _extract_company_name(cls, raw: Dict, company_obj: Dict) -> str:
        return cls._first_text(
            raw.get("company_name"),
            company_obj.get("company_name") if isinstance(company_obj, dict) else None,
            company_obj.get("name") if isinstance(company_obj, dict) else None,
            raw.get("companyName"),
            raw.get("employer_name"),
        )

    @classmethod
    def _extract_location(cls, raw: Dict, company_obj: Dict) -> str:
        direct = cls._first_text(raw.get("location"), raw.get("workLocation"), raw.get("work_location"))
        if direct:
            return direct

        company_city = company_obj.get("city") if isinstance(company_obj, dict) else None
        company_country = company_obj.get("country") if isinstance(company_obj, dict) else None
        parts = [cls._first_text(company_city), cls._first_text(company_country)]
        parts = [part for part in parts if part]
        return ", ".join(parts)

    @classmethod
    def _normalize_salary_value(cls, value: object) -> str:
        if isinstance(value, dict):
            minimum = cls._first_text(value.get("min"), value.get("minimum"), value.get("from"))
            maximum = cls._first_text(value.get("max"), value.get("maximum"), value.get("to"))
            currency = cls._first_text(value.get("currency"))
            if minimum and maximum:
                return f"{minimum} - {maximum}{(' ' + currency) if currency else ''}".strip()
            if minimum:
                return f"Tu {minimum}{(' ' + currency) if currency else ''}".strip()
            if maximum:
                return f"Den {maximum}{(' ' + currency) if currency else ''}".strip()
            return ""
        return cls._first_text(value)

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
                            "company_id": record.company_id,
                            "title": record.title,
                            "company": record.company,
                            "city": record.city,
                            "location": record.location,
                            "salary_min": record.salary_min,
                            "salary_max": record.salary_max,
                            "salary": record.salary,
                            "is_active": record.is_active,
                            "created_at": record.created_at,
                            "category_id": record.category_id,
                            "skills": record.skills,
                            "category": record.category,
                            "work_type": record.work_type,
                            "job_type_id": record.job_type_id,
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
            f"Company ID: {record.company_id}\n"
            f"Title: {record.title}\n"
            f"Company: {record.company}\n"
            f"City: {record.city}\n"
            f"Location: {record.location}\n"
            f"Status: {'active' if record.is_active else 'inactive'}\n"
            f"Created At: {record.created_at}\n"
            f"Updated At: {record.updated_at}\n"
            f"Salary Min: {record.salary_min if record.salary_min is not None else ''}\n"
            f"Salary Max: {record.salary_max if record.salary_max is not None else ''}\n"
            f"Salary: {record.salary}\n"
            f"Category ID: {record.category_id}\n"
            f"Category: {record.category}\n"
            f"Work Type: {record.work_type}\n"
            f"Job Type ID: {record.job_type_id}\n"
            f"Job Type: {record.job_type}\n"
            f"Level: {record.level}\n"
            f"Skills: {record.skills}\n"
            f"Description: {record.description}\n"
            f"Requirements: {record.requirements}\n"
            f"Benefits: {record.benefits}\n"
            f"URL: {record.url}\n"
        )

    @classmethod
    def _extract_salary_bounds(cls, range_value: object, salary_value: object) -> Tuple[Optional[int], Optional[int]]:
        minimum: Optional[int] = None
        maximum: Optional[int] = None

        if isinstance(range_value, dict):
            minimum = cls._to_int(range_value.get("min") or range_value.get("minimum") or range_value.get("from"))
            maximum = cls._to_int(range_value.get("max") or range_value.get("maximum") or range_value.get("to"))

        if minimum is not None or maximum is not None:
            return minimum, maximum

        salary_text = cls._first_text(salary_value)
        if not salary_text:
            return None, None

        numeric_tokens = re.findall(r"\d[\d,._\s]*", salary_text)
        numbers = [cls._to_int(match) for match in numeric_tokens]
        numbers = [value for value in numbers if value is not None]
        if len(numbers) >= 2:
            return numbers[0], numbers[1]
        if len(numbers) == 1:
            return numbers[0], numbers[0]
        return None, None

    @staticmethod
    def _to_int(value: object) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)

        digits = "".join(ch for ch in str(value) if ch.isdigit())
        if not digits:
            return None
        try:
            return int(digits)
        except ValueError:
            return None

    @staticmethod
    def _to_bool(value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        lowered = str(value or "").strip().lower()
        if lowered in {"true", "1", "yes", "y", "active"}:
            return True
        if lowered in {"false", "0", "no", "n", "inactive"}:
            return False
        return False

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
