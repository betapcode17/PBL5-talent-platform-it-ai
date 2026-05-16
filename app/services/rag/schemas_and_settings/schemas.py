"""Core data models moved under schemas_and_settings package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class JobRecord:
    job_id: str = ""
    title: str = ""
    company_id: str = ""
    company: str = ""
    city: str = ""
    location: str = ""
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None
    salary: str = ""
    is_active: bool = True
    created_at: str = ""
    category_id: str = ""
    skills: str = ""
    category: str = ""
    job_type_id: str = ""
    description: str = ""
    requirements: str = ""
    benefits: str = ""
    work_type: str = ""
    job_type: str = ""
    level: str = ""
    url: str = ""
    updated_at: str = ""


@dataclass(frozen=True)
class CompanyRecord:
    company_id: str = ""
    name: str = ""
    industry: str = ""
    company_type: str = ""
    size: str = ""
    location: str = ""
    website: str = ""
    email: str = ""
    description: str = ""
    key_skills: str = ""
    why_join: str = ""
    updated_at: str = ""


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    

    
    @property
    def id(self) -> str:
        """Backward-compatible alias for older code that accessed `chunk.id`.

        Prefer `chunk.chunk_id` in new code.
        """
        return self.chunk_id


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    distance: float
    rerank_score: float
    source_key: str
    cross_encoder_score: Optional[float] = None


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    items: List[RetrievedChunk]
    latency_ms: float
    cached: bool = False


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    prompt_tokens_estimate: int
    completion_tokens_estimate: int
    model: str
    used_cuda: bool
    dtype: str


@dataclass(frozen=True)
class SyncResult:
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
