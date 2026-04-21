"""
Reverse matching: Find candidates for a job using RAG on CV index.

NOTE: This module requires Gemini API which has been removed.
Currently disabled - use rag_matching.py for job matching instead.
"""

import json
import logging
from typing import List, Dict, Any
from langchain_core.documents import Document # type: ignore

from app.models.core import CandidateSearchInput, CandidateSearchResponse, MatchedCandidate, Suggestion
from .chroma_utils import get_vectorstore
from .db_utils import get_db_connection
from .rag_helpers import _prefix_doc_with_id

logging.basicConfig(level=logging.INFO)

async def match_candidates_to_job(input: CandidateSearchInput) -> CandidateSearchResponse:
    """
    Disabled: Reverse match requires Gemini API which has been removed.
    Use rag_matching.py for job matching instead.
    """
    logging.warning("⚠ Candidate matching requires Gemini API which has been removed.")
    return CandidateSearchResponse(
        total=0,
        candidates=[],
        suggestions=[Suggestion(
            skill_or_experience="N/A",
            suggestion="Candidate matching feature requires Gemini API. Use job matching instead (POST /matching/match-jobs)"
        )]
    )