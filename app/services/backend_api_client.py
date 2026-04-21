# app/services/backend_api_client.py
"""
Backend API Client - Handles communication with NestJS backend
"""

import logging
import httpx # type: ignore
from typing import Dict, Any, Optional, List
from app.config import BACKEND_API_URL

logger = logging.getLogger(__name__)


class BackendAPIClient:
    """Client for calling NestJS backend APIs"""
    
    def __init__(self, base_url: str = BACKEND_API_URL, timeout: float = 30.0):
        """
        Initialize Backend API Client
        
        Args:
            base_url: Base URL of the NestJS backend (e.g., http://localhost:4000)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        logger.info(f" BackendAPIClient initialized with base_url: {self.base_url}")
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Make HTTP request to backend
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (e.g., /jobs/search)
            params: Query parameters
            json_data: JSON body data
            headers: Custom headers
            
        Returns:
            Response JSON
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json_data,
                    headers=headers or {}
                )
                
                # Log request
                logger.debug(f"[Backend API] {method} {endpoint} -> Status {response.status_code}")
                
                # Handle errors
                if response.status_code >= 400:
                    logger.error(
                        f"Backend API error: {response.status_code} - {response.text}"
                    )
                    return {
                        "error": f"Backend error: {response.status_code}",
                        "status": "failed",
                        "details": response.text[:200]
                    }
                
                return response.json()
        
        except httpx.TimeoutException:
            logger.error(f"Backend API timeout for {endpoint}")
            return {
                "error": "Request timeout",
                "status": "failed"
            }
        except httpx.ConnectError:
            logger.error(f"Cannot connect to backend at {self.base_url}")
            return {
                "error": "Cannot connect to backend",
                "status": "failed"
            }
        except Exception as e:
            logger.error(f"Backend API request failed: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }
    
    async def search_jobs(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        location: Optional[str] = None,
        salary_min: Optional[str] = None,
        page: int = 1,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """
        Search jobs from backend
        
        Args:
            query: Search query string (optional)
            category: Job category filter
            location: Location filter
            salary_min: Minimum salary filter
            page: Page number (1-indexed)
            limit: Results per page
            
        Returns:
            {
                "jobs": [...],
                "total": <int>,
                "page": <int>,
                "limit": <int>
            }
        """
        # Decide which endpoint to use based on filters
        has_filters = category or location or salary_min
        has_query = query and query.strip()
        
        params = {
            "page": page,
            "limit": limit,
        }
        
        # If no query and no filters, use /jobs endpoint (list all jobs)
        if not has_query and not has_filters:
            endpoint = "/jobs"
        else:
            # Use /jobs/search for filtered queries
            endpoint = "/jobs/search"
            
            # Only add query if not empty
            if has_query:
                params["q"] = query
        
        # Add filters
        if category:
            params["category"] = category
        if location:
            params["location"] = location
        if salary_min:
            params["salaryMin"] = salary_min
        
        result = await self._make_request("GET", endpoint, params=params)
        
        # Transform response if needed
        if "error" not in result:
            # Backend returns structured data, pass through
            logger.info(f"Search found {result.get('total', 0)} jobs")
        
        return result
    
    async def get_job_details(self, job_id: int) -> Dict[str, Any]:
        """
        Get job details from backend
        
        Args:
            job_id: Job ID
            
        Returns:
            Job details object
        """
        result = await self._make_request("GET", f"/jobs/{job_id}")
        
        if "error" in result:
            logger.warning(f"Failed to get job details for job_id={job_id}")
        
        return result
    
    async def get_company_jobs(
        self,
        company_id: int,
        page: int = 1,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """
        Get jobs from specific company
        
        Args:
            company_id: Company ID
            page: Page number
            limit: Results per page
            
        Returns:
            Jobs list and pagination info
        """
        params = {
            "page": page,
            "limit": limit,
        }
        
        result = await self._make_request(
            "GET",
            f"/jobs/company/{company_id}",
            params=params
        )
        
        return result
    
    async def get_all_jobs(
        self,
        page: int = 1,
        limit: int = 20,
        active_only: bool = True,
    ) -> Dict[str, Any]:
        """
        Get all jobs with optional filters
        
        Args:
            page: Page number
            limit: Results per page
            active_only: Only active jobs
            
        Returns:
            Jobs list and pagination info
        """
        params = {
            "page": page,
            "limit": limit,
            "active": active_only,
        }
        
        result = await self._make_request("GET", "/jobs", params=params)
        
        return result
    
    async def create_job(
        self,
        title: str,
        description: str,
        category_id: int,
        job_type_id: int,
        company_id: int,
        salary_range: Dict[str, int],
        requirements: List[str],
    ) -> Dict[str, Any]:
        """
        Create new job posting (admin only)
        
        Args:
            title: Job title
            description: Job description
            category_id: Category ID
            job_type_id: Job type ID
            company_id: Company ID
            salary_range: {"min": <int>, "max": <int>}
            requirements: List of requirements
            
        Returns:
            Created job object
        """
        json_data = {
            "title": title,
            "description": description,
            "categoryId": category_id,
            "jobTypeId": job_type_id,
            "companyId": company_id,
            "salaryRange": salary_range,
            "requirements": requirements,
        }
        
        result = await self._make_request("POST", "/jobs", json_data=json_data)
        
        return result


# Singleton instance
_backend_client: Optional[BackendAPIClient] = None


def get_backend_client() -> BackendAPIClient:
    """Get or create backend API client"""
    global _backend_client
    if _backend_client is None:
        _backend_client = BackendAPIClient()
    return _backend_client
