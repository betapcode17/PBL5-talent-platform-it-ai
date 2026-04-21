"""
Matching Tools
Handles CV-to-job matching and compatibility analysis
"""

import logging
from typing import Optional
import httpx

logger = logging.getLogger(__name__)


class MatchingTools:
    """CV-to-job matching tools"""
    
    def __init__(self, http_client: httpx.AsyncClient, headers: dict):
        self.http_client = http_client
        self.headers = headers
    
    async def match_cv_to_job(self, cv_id: str, job_id: str) -> dict:
        """Match a CV against a job posting and get compatibility score"""
        try:
            response = await self.http_client.post(
                "/matching/match",
                json={"cv_id": cv_id, "job_id": job_id},
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error matching CV to job: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def find_best_matches(self, cv_id: str, limit: int = 5) -> dict:
        """Find the best job matches for a given CV"""
        try:
            response = await self.http_client.get(
                f"/matching/best-matches/{cv_id}",
                params={"limit": limit},
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error finding best matches: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def get_match_explanation(self, cv_id: str, job_id: str) -> dict:
        """Get detailed explanation of why a CV matches a job"""
        try:
            response = await self.http_client.get(
                f"/matching/explain",
                params={"cv_id": cv_id, "job_id": job_id},
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting match explanation: {e}")
            return {"error": str(e), "status": "failed"}


# Tool definitions for registration
MATCHING_TOOLS = [
    {
        "name": "match_cv_to_job",
        "description": "Match a CV against a job posting and get compatibility score",
        "inputSchema": {
            "type": "object",
            "properties": {
                "cv_id": {
                    "type": "string",
                    "description": "The ID of the CV"
                },
                "job_id": {
                    "type": "string",
                    "description": "The ID of the job"
                }
            },
            "required": ["cv_id", "job_id"]
        }
    },
    {
        "name": "find_best_matches",
        "description": "Find the best job matches for a given CV",
        "inputSchema": {
            "type": "object",
            "properties": {
                "cv_id": {
                    "type": "string",
                    "description": "The ID of the CV"
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of top matches to return (default: 5)",
                    "default": 5
                }
            },
            "required": ["cv_id"]
        }
    },
    {
        "name": "get_match_explanation",
        "description": "Get detailed explanation of why a CV matches a job",
        "inputSchema": {
            "type": "object",
            "properties": {
                "cv_id": {
                    "type": "string",
                    "description": "The ID of the CV"
                },
                "job_id": {
                    "type": "string",
                    "description": "The ID of the job"
                }
            },
            "required": ["cv_id", "job_id"]
        }
    },
]
