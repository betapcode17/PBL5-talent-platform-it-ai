"""
Candidate Tools
Handles candidate profile management
"""

import logging
from typing import Optional
import httpx

logger = logging.getLogger(__name__)


class CandidateTools:
    """Candidate management tools"""
    
    def __init__(self, http_client: httpx.AsyncClient, headers: dict):
        self.http_client = http_client
        self.headers = headers
    
    async def create_candidate(self, name: str, email: str, phone: Optional[str] = None, 
                              skills: Optional[list] = None) -> dict:
        """Create a new candidate profile"""
        try:
            response = await self.http_client.post(
                "/candidates",
                json={
                    "name": name,
                    "email": email,
                    "phone": phone,
                    "skills": skills or []
                },
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error creating candidate: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def get_candidate(self, candidate_id: str) -> dict:
        """Get candidate profile information"""
        try:
            response = await self.http_client.get(
                f"/candidates/{candidate_id}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting candidate: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def list_candidates(self, skip: int = 0, limit: int = 20, 
                             skill: Optional[str] = None) -> dict:
        """List all candidates with optional filtering"""
        try:
            params = {"skip": skip, "limit": limit}
            if skill:
                params["skill"] = skill
            
            response = await self.http_client.get(
                "/candidates",
                params=params,
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error listing candidates: {e}")
            return {"error": str(e), "status": "failed"}


# Tool definitions for registration
CANDIDATE_TOOLS = [
    {
        "name": "create_candidate",
        "description": "Create a new candidate profile",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Candidate name"
                },
                "email": {
                    "type": "string",
                    "description": "Candidate email"
                },
                "phone": {
                    "type": "string",
                    "description": "Candidate phone number"
                },
                "skills": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of candidate skills"
                }
            },
            "required": ["name", "email"]
        }
    },
    {
        "name": "get_candidate",
        "description": "Get candidate profile information",
        "inputSchema": {
            "type": "object",
            "properties": {
                "candidate_id": {
                    "type": "string",
                    "description": "The ID of the candidate"
                }
            },
            "required": ["candidate_id"]
        }
    },
    {
        "name": "list_candidates",
        "description": "List all candidates with optional filtering",
        "inputSchema": {
            "type": "object",
            "properties": {
                "skip": {
                    "type": "integer",
                    "description": "Number to skip (default: 0)",
                    "default": 0
                },
                "limit": {
                    "type": "integer",
                    "description": "Number to return (default: 20)",
                    "default": 20
                },
                "skill": {
                    "type": "string",
                    "description": "Filter by specific skill"
                }
            },
            "required": []
        }
    },
]
