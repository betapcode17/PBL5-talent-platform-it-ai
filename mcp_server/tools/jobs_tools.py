"""
Jobs Tools
Handles all job search and listing related MCP tools
"""

import logging
from typing import Optional
import httpx # type: ignore

logger = logging.getLogger(__name__)


class JobsTools:
    """Jobs search and listing tools"""
    
    def __init__(self, http_client: httpx.AsyncClient, headers: dict):
        self.http_client = http_client
        self.headers = headers
    
    async def search_jobs(
        self,
        query: str,
        limit: int = 20,
        category: Optional[str] = None,
        location: Optional[str] = None,
        salary_min: Optional[str] = None,
        page: int = 1
    ) -> dict:
        """
        Search for jobs by keywords, skills, or location
        
        Args:
            query: Search keywords (title, description, requirements, location)
            limit: Results per page (default: 20, max: 100)
            category: Filter by job category
            location: Filter by location
            salary_min: Minimum salary (e.g., "10k", "1m", "50", "1b")
            page: Page number (default: 1)
        """
        try:
            params = {
                "q": query,
                "limit": min(limit, 100),
                "page": max(page, 1)
            }
            
            if category:
                params["category"] = category
            if location:
                params["location"] = location
            if salary_min:
                params["salaryMin"] = salary_min
            
            response = await self.http_client.get(
                "/jobs/search",
                params=params,
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error searching jobs: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def get_job_details(self, job_id: int) -> dict:
        """
        Get detailed information about a specific job posting
        
        Args:
            job_id: The ID of the job
        
        Returns:
            Job details including company, category, requirements, salary range, etc.
        """
        try:
            response = await self.http_client.get(
                f"/jobs/{job_id}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting job details: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def list_all_jobs(self, page: int = 1, limit: int = 20, active: Optional[bool] = None) -> dict:
        """
        List all available jobs with pagination
        
        Args:
            page: Page number (default: 1, must be >= 1)
            limit: Jobs per page (default: 20, must be 1-100)
            active: Filter by active status (True/False/None for all)
        
        Returns:
            List of jobs with pagination info and filters
        """
        try:
            if page < 1:
                return {"error": "page must be >= 1", "status": "failed"}
            if limit < 1 or limit > 100:
                return {"error": "limit must be between 1 and 100", "status": "failed"}
            
            params = {
                "page": page,
                "limit": limit
            }
            
            if isinstance(active, bool):
                params["active"] = active
            
            response = await self.http_client.get(
                "/jobs",
                params=params,
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error listing jobs: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def get_company_jobs(
        self,
        company_id: int,
        page: int = 1,
        limit: int = 10,
        active: Optional[bool] = None
    ) -> dict:
        """
        Get all jobs posted by a specific company
        
        Args:
            company_id: The company ID
            page: Page number (default: 1)
            limit: Jobs per page (default: 10, max: 100)
            active: Filter by active status
        
        Returns:
            List of company jobs with pagination
        """
        try:
            if page < 1:
                return {"error": "page must be >= 1", "status": "failed"}
            if limit < 1 or limit > 100:
                return {"error": "limit must be between 1 and 100", "status": "failed"}
            
            params = {
                "page": page,
                "limit": limit
            }
            
            if isinstance(active, bool):
                params["active"] = active
            
            response = await self.http_client.get(
                f"/jobs/company/{company_id}",
                params=params,
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting company jobs: {e}")
            return {"error": str(e), "status": "failed"}


# Tool definitions for registration
JOBS_TOOLS = [
    {
        "name": "search_jobs",
        "description": "🔍 SEARCH JOBS from database. USE THIS when user asks: 'tìm việc', 'find job', 'search jobs'. Returns matching jobs with title, company, salary, location, skills. Format results with company name, salary range, and key requirements.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Main search keywords: job title (e.g., 'Backend', 'Frontend', 'DevOps') OR skills OR job description. REQUIRED."
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of results to return (default: 20, max: 100). Use 10-15 for concise results.",
                    "default": 20
                },
                "page": {
                    "type": "integer",
                    "description": "Pagination page number (starts from 1). Use page=1 for initial search.",
                    "default": 1
                },
                "category": {
                    "type": "string",
                    "description": "Filter by job category name (e.g., 'Software', 'Data Science', 'Finance'). OPTIONAL."
                },
                "location": {
                    "type": "string",
                    "description": "Filter by city/location (e.g., 'Hà Nội', 'TPHCM', 'Đà Nẵng'). OPTIONAL."
                },
                "salary_min": {
                    "type": "string",
                    "description": "Minimum salary filter in millions (e.g., '10m', '50m'). Supports k/m/b suffixes. OPTIONAL."
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_job_details",
        "description": "📋 GET DETAILED JOB INFO. Use job_id from search_jobs results to fetch complete job posting. Returns: full description, all requirements, salary details, benefits, company info, work conditions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "integer",
                    "description": "The unique job ID from search_jobs results. Find this in the search output and use it to get full details."
                }
            },
            "required": ["job_id"]
        }
    },
    {
        "name": "list_all_jobs",
        "description": "List all available jobs with pagination support. Can filter by active status.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "page": {
                    "type": "integer",
                    "description": "Page number starting from 1 (default: 1, must be >= 1)",
                    "default": 1
                },
                "limit": {
                    "type": "integer",
                    "description": "Jobs per page (default: 20, must be 1-100)",
                    "default": 20
                },
                "active": {
                    "type": "boolean",
                    "description": "Filter by active status (optional: true for active, false for inactive, null for all)"
                }
            },
            "required": []
        }
    },
    {
        "name": "get_company_jobs",
        "description": "Get all jobs posted by a specific company with pagination",
        "inputSchema": {
            "type": "object",
            "properties": {
                "company_id": {
                    "type": "integer",
                    "description": "The company ID"
                },
                "page": {
                    "type": "integer",
                    "description": "Page number (default: 1, must be >= 1)",
                    "default": 1
                },
                "limit": {
                    "type": "integer",
                    "description": "Jobs per page (default: 10, must be 1-100)",
                    "default": 10
                },
                "active": {
                    "type": "boolean",
                    "description": "Filter by active status (optional)"
                }
            },
            "required": ["company_id"]
        }
    },
]
