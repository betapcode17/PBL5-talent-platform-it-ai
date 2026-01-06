# tests/test_rag_matching.py
"""
Unit tests for RAG matching components.
Run with: poetry run pytest tests/test_rag_matching.py
"""

import pytest
import asyncio
import json
from services.rag_matching import match_cv, get_rag_components
from services.rag_helpers import _to_int_job_id, verify_job_id_consistency, _prefix_doc_with_id
from langchain_core.documents import Document
from unittest.mock import Mock, patch

@pytest.fixture
def sample_cv():
    return {
        "cv_id": 1,
        "skills": ["Python", "SQL"],
        "aspirations": "Senior Data Scientist",
        "experience": "Data Analyst at Tech Corp",
        "education": "Bachelor in CS"
    }

@pytest.fixture
def sample_filtered_ids():
    return [1, 2, 3]

@pytest.mark.asyncio
async def test_match_cv(sample_cv, sample_filtered_ids):
    with patch('app.services.chroma_utils.get_vectorstore') as mock_vs, \
         patch('app.services.rag_matching.get_rag_components') as mock_components:
        # Mock vectorstore
        mock_retriever = Mock()
        mock_retriever.get_relevant_documents.return_value = [Document(page_content="Sample job", metadata={"job_id": "1"})]
        mock_vs.return_value.as_retriever.return_value = mock_retriever
        
        # Mock components
        mock_retriever_mock, mock_qa_chain, _ = mock_components.return_value
        mock_qa_chain.ainvoke.return_value = {
            "matched_jobs": [{"job_id": "1", "match_score": 0.8}],
            "suggestions": [{"skill_or_experience": "ML", "suggestion": "Learn ML"}]
        }
        
        result = await match_cv(sample_cv, sample_filtered_ids, "test_session")
        
        assert result["cv_id"] == 1
        assert len(result["matched_jobs"]) == 1
        assert "suggestions" in result
        mock_qa_chain.ainvoke.assert_called_once()

def test_to_int_job_id():
    assert _to_int_job_id(123) == 123
    assert _to_int_job_id("job_456") == 456
    assert _to_int_job_id("invalid") is None

def test_prefix_doc_with_id():
    doc = Document(page_content="Long content...", metadata={"job_id": "789", "job_title": "Dev", "job_url": "url.com"})
    prefixed = _prefix_doc_with_id(doc)
    assert "JOB_ID: 789" in prefixed.page_content
    assert len(prefixed.page_content) < len("Long content...") + 100  # Shortened

@pytest.mark.asyncio
async def test_verify_job_id_consistency():
    # This requires DB setup; mock for test
    with patch('app.services.db_utils.get_db_connection') as mock_db, \
         patch('app.services.chroma_utils.get_vectorstore') as mock_chroma:
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchone.return_value = {"job_url": "test.com", "job_title": "Test", "work_location": "HN", "skills": "[]"}
        mock_conn.cursor.return_value = mock_cursor
        mock_db.return_value.__enter__.return_value = mock_conn
        
        mock_vs = mock_chroma.return_value
        mock_vs.get.return_value = {"ids": ["id"], "metadatas": [{"job_id": "1", "job_url": "test.com", "job_title": "Test", "work_location": "HN", "skills": "[]"}]}
        
        result = verify_job_id_consistency(1)
        assert result is True  # Assuming consistency in mock

if __name__ == "__main__":
    # For manual testing
    asyncio.run(test_match_cv(sample_cv, []))  # Adjust as needed