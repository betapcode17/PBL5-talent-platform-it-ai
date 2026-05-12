"""Core data models for the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class JobRecord:
    """Normalized job payload fetched from backend API."""

    job_id: str
    title: str
    company: str
    location: str
    salary: str
    skills: str
    category: str
    description: str
    requirements: str
    benefits: str
    work_type: str
    job_type: str
    level: str
    url: str
    updated_at: str


@dataclass(frozen=True)
class CompanyRecord:
    """Normalized company payload fetched from backend API."""

    company_id: str
    name: str
    industry: str
    company_type: str
    size: str
    location: str
    website: str
    email: str
    description: str
    key_skills: str
    why_join: str
    updated_at: str


@dataclass(frozen=True)
class ChunkRecord:
    """Text chunk used for embedding and vector indexing."""

    chunk_id: str
    text: str
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class RetrievedChunk:
    """Retrieved chunk returned by vector search."""

    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    distance: float
    rerank_score: float
    source_key: str


@dataclass(frozen=True)
class RetrievalResult:
    """Final retrieval output consumed by generation step."""

    query: str
    items: List[RetrievedChunk]
    latency_ms: float
    cached: bool = False


@dataclass(frozen=True)
class GenerationResult:
    """Model generation output for chat response."""

    answer: str
    prompt_tokens_estimate: int
    completion_tokens_estimate: int
    model: str
    used_cuda: bool
    dtype: str


@dataclass(frozen=True)
class SyncResult:
    """Result for an ingestion/sync run."""

    fetched_jobs: int
    indexed_chunks: int
    skipped_jobs: int
    backend_pages: int
    took_seconds: float
    fetched_companies: int = 0
    indexed_job_chunks: int = 0
    indexed_company_chunks: int = 0
    stale_chunks_removed: int = 0
    errors: Optional[List[str]] = None
    warnings: List[str] = field(default_factory=list)
