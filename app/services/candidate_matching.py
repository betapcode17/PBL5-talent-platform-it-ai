# app/services/candidate_matching.py
"""Placeholder candidate matching service used by /candidates endpoint.
Replace with real implementation later.
"""
from app.models.core import MatchedCandidate, CandidateSearchInput
from typing import Dict, Any


async def match_candidates_to_job(input: CandidateSearchInput):
    # Minimal stub: return empty list
    return {
        "total": 0,
        "candidates": [],
        "suggestions": []
    }
