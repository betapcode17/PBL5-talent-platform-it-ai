from fastapi import APIRouter

from models.core import CandidateSearchInput, CandidateSearchResponse
from services.candidate_matching import match_candidates_to_job 

router = APIRouter(prefix="/candidates", tags=["Candidates"])  

@router.post("/search", response_model=CandidateSearchResponse)
async def search_candidates(input: CandidateSearchInput):
    """Tìm CVs matching với job description."""
    result = await match_candidates_to_job(input)
    return result