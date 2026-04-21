"""
CV Analysis Tools
Handles all CV-related MCP tools
"""

import logging
from typing import Optional
import httpx # type: ignore

logger = logging.getLogger(__name__)


class CVTools:
    """CV Analysis tools"""
    
    def __init__(self, http_client: httpx.AsyncClient, headers: dict):
        self.http_client = http_client
        self.headers = headers
    
    async def analyze_cv(self, cv_text: str) -> dict:
        """Analyze CV content and extract key information"""
        try:
            response = await self.http_client.post(
                "/cv/analyze",
                json={"cv_text": cv_text},
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error analyzing CV: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def get_cv_insights(self, cv_id: str) -> dict:
        """Get AI-generated insights about a CV"""
        try:
            response = await self.http_client.get(
                f"/cv/{cv_id}/insights",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting CV insights: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def generate_cv_improvements(self, cv_id: str) -> dict:
        """Generate specific improvement suggestions for a CV"""
        try:
            response = await self.http_client.post(
                f"/cv/{cv_id}/improve",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error generating CV improvements: {e}")
            return {"error": str(e), "status": "failed"}


# Tool definitions for registration
CV_TOOLS = [
    {
        "name": "analyze_cv",
        "description": "Analyze a CV file and extract key information (skills, experience, education, red flags)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "cv_text": {
                    "type": "string",
                    "description": "The CV content as text"
                }
            },
            "required": ["cv_text"]
        }
    },
    {
        "name": "get_cv_insights",
        "description": "Get AI-generated insights about a CV (strengths, gaps, recommendations)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "cv_id": {
                    "type": "string",
                    "description": "The ID of the CV to analyze"
                }
            },
            "required": ["cv_id"]
        }
    },
    {
        "name": "generate_cv_improvements",
        "description": "Generate specific improvement suggestions for a CV",
        "inputSchema": {
            "type": "object",
            "properties": {
                "cv_id": {
                    "type": "string",
                    "description": "The ID of the CV to improve"
                }
            },
            "required": ["cv_id"]
        }
    },
]
