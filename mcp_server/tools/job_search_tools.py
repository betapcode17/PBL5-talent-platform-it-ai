"""
Job Search Tools - Call Backend NestJS API for job listings
Handles job search, filters, and fetching all jobs
"""

import logging
from typing import Optional, List, Dict, Any
import httpx

logger = logging.getLogger(__name__)

# Backend API base URL
BACKEND_API_URL = "http://127.0.0.1:3000"


class JobSearchTools:
    """Tools for searching jobs via Backend NestJS API"""
    
    def __init__(self, http_client: httpx.AsyncClient = None, headers: dict = None):
        self.http_client = http_client
        self.headers = headers or {"Content-Type": "application/json"}
        self.backend_url = BACKEND_API_URL
    
    async def search_jobs(
        self,
        keyword: Optional[str] = None,
        category: Optional[str] = None,
        location: Optional[str] = None,
        salary_min: Optional[str] = None,
        page: int = 1,
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        Search for jobs with filters
        
        Args:
            keyword: Search keyword (e.g., "Backend", "Python")
            category: Job category (e.g., "web", "mobile")
            location: Location filter (e.g., "Ha Noi", "TP.HCM")
            salary_min: Minimum salary (e.g., "20M", "15M")
            page: Page number (default 1)
            limit: Results per page (default 20)
        
        Returns:
            Dict with jobs, total count, and filters
        """
        try:
            # Build query parameters
            params = {
                "page": page,
                "limit": min(limit, 100),  # Cap at 100
            }
            
            if keyword:
                params["q"] = keyword
            if category:
                params["category"] = category
            if location:
                params["location"] = location
            if salary_min:
                params["salaryMin"] = salary_min
            
            logger.info(f"Searching jobs with params: {params}")
            
            # Make request to backend
            response = await self.http_client.get(
                f"{self.backend_url}/jobs/search",
                params=params,
                headers=self.headers,
                timeout=30.0
            )
            
            response.raise_for_status()
            data = response.json()
            
            logger.info(f"Found {len(data.get('jobs', []))} jobs")
            return {
                "success": True,
                "jobs": data.get("jobs", []),
                "total": data.get("total", 0),
                "page": page,
                "limit": limit,
                "filters": data.get("filters", {}),
                "status": "ok"
            }
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Backend API error {e.response.status_code}: {e.response.text}")
            return {
                "success": False,
                "error": f"Backend error: {e.response.status_code}",
                "jobs": [],
                "total": 0,
                "status": "error"
            }
        except Exception as e:
            logger.error(f"Error searching jobs: {e}")
            return {
                "success": False,
                "error": str(e),
                "jobs": [],
                "total": 0,
                "status": "error"
            }
    
    async def get_all_jobs(
        self,
        page: int = 1,
        limit: int = 20,
        active: Optional[bool] = True
    ) -> Dict[str, Any]:
        """
        Get all jobs without filters
        
        Args:
            page: Page number (default 1)
            limit: Results per page (default 20)
            active: Filter by active status (default True)
        
        Returns:
            Dict with all jobs and metadata
        """
        try:
            params = {
                "page": page,
                "limit": min(limit, 100),
            }
            
            if active is not None:
                params["active"] = str(active).lower()
            
            logger.info(f"Fetching all jobs with params: {params}")
            
            response = await self.http_client.get(
                f"{self.backend_url}/jobs",
                params=params,
                headers=self.headers,
                timeout=30.0
            )
            
            response.raise_for_status()
            data = response.json()
            
            logger.info(f"Retrieved {len(data.get('jobs', []))} jobs (total: {data.get('total', 0)})")
            return {
                "success": True,
                "jobs": data.get("jobs", []),
                "total": data.get("total", 0),
                "page": page,
                "limit": limit,
                "status": "ok"
            }
            
        except Exception as e:
            logger.error(f"Error fetching all jobs: {e}")
            return {
                "success": False,
                "error": str(e),
                "jobs": [],
                "total": 0,
                "status": "error"
            }
    
    async def get_job_detail(self, job_id: int) -> Dict[str, Any]:
        """
        Get detailed information about a specific job
        
        Args:
            job_id: ID of the job post
        
        Returns:
            Dict with job details
        """
        try:
            logger.info(f"Fetching job details for ID: {job_id}")
            
            response = await self.http_client.get(
                f"{self.backend_url}/jobs/{job_id}",
                headers=self.headers,
                timeout=30.0
            )
            
            response.raise_for_status()
            job = response.json()
            
            logger.info(f"Retrieved job: {job.get('title', 'Unknown')}")
            return {
                "success": True,
                "job": job,
                "status": "ok"
            }
            
        except Exception as e:
            logger.error(f"Error fetching job detail: {e}")
            return {
                "success": False,
                "error": str(e),
                "status": "error"
            }
    
    def format_jobs_response(self, jobs: List[Dict[str, Any]]) -> str:
        """Format jobs list into readable text for LLM"""
        if not jobs:
            return "Không tìm thấy công việc phù hợp."
        
        formatted = []
        for i, job in enumerate(jobs[:10], 1):  # Show top 10
            title = job.get("title", "N/A")
            company = job.get("company", {}).get("company_name", "N/A")
            location = job.get("location", "N/A")
            salary = job.get("salary", "Thương lượng")
            
            formatted.append(
                f"{i}. **{title}** ({company})\n"
                f"   📍 {location} | 💰 {salary}"
            )
        
        if len(jobs) > 10:
            formatted.append(f"\n...và {len(jobs) - 10} công việc khác")
        
        return "\n".join(formatted)


# Tool definitions for registration in MCP
JOB_SEARCH_TOOLS = [
    {
        "name": "search_jobs",
        "description": "Search for jobs in the database with optional filters (keyword, location, category, salary)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "Search keyword (e.g., 'Backend', 'Python', 'Frontend')"
                },
                "category": {
                    "type": "string",
                    "description": "Job category filter (e.g., 'web', 'mobile', 'data')"
                },
                "location": {
                    "type": "string",
                    "description": "Location filter (e.g., 'Ha Noi', 'TP.HCM', 'Da Nang')"
                },
                "salary_min": {
                    "type": "string",
                    "description": "Minimum salary filter (e.g., '20M', '15M', '10 triệu')"
                },
                "page": {
                    "type": "integer",
                    "description": "Page number for pagination (default 1)",
                    "default": 1
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of results per page (default 20, max 100)",
                    "default": 20
                }
            }
        }
    },
    {
        "name": "get_all_jobs",
        "description": "Retrieve all available jobs without specific filters",
        "inputSchema": {
            "type": "object",
            "properties": {
                "page": {
                    "type": "integer",
                    "description": "Page number for pagination (default 1)",
                    "default": 1
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of results per page (default 20, max 100)",
                    "default": 20
                },
                "active": {
                    "type": "boolean",
                    "description": "Filter by active status (default true)",
                    "default": True
                }
            }
        }
    },
    {
        "name": "get_job_detail",
        "description": "Get detailed information about a specific job post",
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "integer",
                    "description": "The ID of the job post"
                }
            },
            "required": ["job_id"]
        }
    }
]
