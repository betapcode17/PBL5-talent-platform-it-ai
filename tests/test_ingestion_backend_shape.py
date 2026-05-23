from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.services.rag.core.ingestion import IngestionService


@patch("app.services.rag.core.ingestion.get_backend_client")
def test_normalize_job_payload_drops_skills(mock_backend_client) -> None:
    service = IngestionService(MagicMock(), MagicMock())

    normalized = service._normalize_job_payload(
        {
            "id": 123,
            "title": "Backend Engineer",
            "company": "TechCorp Vietnam",
            "description": "Build APIs",
            "skills": ["Python", "Docker"],
            "technologies": ["NestJS"],
            "requirements": "FastAPI",
            "location": "Ho Chi Minh City",
            "salary": "40-60m",
            "category": "IT",
            "job_type": "Toàn thời gian",
            "updatedDate": "2026-04-19T14:23:28.203Z",
        }
    )

    assert "skills" not in normalized
    assert normalized["title"] == "Backend Engineer"
    assert normalized["company"] == "TechCorp Vietnam"
    assert normalized["location"] == "Ho Chi Minh City"
    assert normalized["job_type"] == "Toàn thời gian"


@patch("app.services.rag.core.ingestion.get_backend_client")
def test_chunk_job_records_do_not_inject_skills(mock_backend_client) -> None:
    service = IngestionService(MagicMock(), MagicMock())

    chunks = service.chunk_job_records(
        {
            "id": 123,
            "title": "Backend Engineer",
            "company": "TechCorp Vietnam",
            "description": "Build APIs and services",
            "location": "Ho Chi Minh City",
            "salary": "40-60m",
            "category": "IT",
            "job_type": "Toàn thời gian",
            "updated_at": "2026-04-19T14:23:28.203Z",
        }
    )

    assert chunks
    chunk = chunks[0]
    assert "Skills:" not in chunk.text
    assert "skills" not in chunk.metadata
    assert "Job Type:" in chunk.text
    assert "Location:" in chunk.text
