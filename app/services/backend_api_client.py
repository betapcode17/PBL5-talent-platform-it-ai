# app/services/backend_api_client.py
"""Backend API client utilities for the NestJS backend."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx  # type: ignore

from app.config import BACKEND_API_URL

logger = logging.getLogger(__name__)


class BackendAPIClient:
    """Client for calling NestJS backend APIs."""

    def __init__(self, base_url: str = BACKEND_API_URL, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        logger.info("BackendAPIClient initialized with base_url=%s", self.base_url)

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout, connect=min(10.0, self.timeout)),
                follow_redirects=True,
            )
        return self._client

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"

        try:
            response = await self._get_client().request(
                method=method,
                url=url,
                params=params,
                json=json_data,
                headers=headers or {},
            )
            logger.debug("[Backend API] %s %s -> %s", method, endpoint, response.status_code)

            if response.status_code >= 400:
                logger.error("Backend API error %s for %s: %s", response.status_code, endpoint, response.text)
                return {
                    "error": f"Backend error: {response.status_code}",
                    "status": "failed",
                    "details": response.text[:500],
                }

            return response.json()
        except httpx.TimeoutException:
            logger.error("Backend API timeout for %s", endpoint)
            return {"error": "Request timeout", "status": "failed"}
        except httpx.ConnectError:
            logger.error("Cannot connect to backend at %s", self.base_url)
            return {"error": "Cannot connect to backend", "status": "failed"}
        except Exception as e:
            logger.error("Backend API request failed for %s: %s", endpoint, e)
            return {"error": str(e), "status": "failed"}

    async def search_jobs(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        location: Optional[str] = None,
        salary_min: Optional[str] = None,
        page: int = 1,
        limit: int = 20,
    ) -> Dict[str, Any]:
        has_filters = category or location or salary_min
        has_query = query and query.strip()

        params = {
            "page": page,
            "limit": limit,
        }

        if not has_query and not has_filters:
            endpoint = "/jobs"
        else:
            endpoint = "/jobs/search"
            if has_query:
                params["q"] = query # type: ignore

        if category:
            params["category"] = category # type: ignore
        if location:
            params["location"] = location # type: ignore
        if salary_min:
            params["salaryMin"] = salary_min # type: ignore

        result = await self._make_request("GET", endpoint, params=params)
        if "error" not in result:
            logger.info("Search found %s jobs", result.get("total", 0))

        return result

    async def get_job_details(self, job_id: int) -> Dict[str, Any]:
        result = await self._make_request("GET", f"/jobs/{job_id}")
        if "error" in result:
            logger.warning("Failed to get job details for job_id=%s", job_id)
        return result

    async def get_company_jobs(
        self,
        company_id: int,
        page: int = 1,
        limit: int = 10,
    ) -> Dict[str, Any]:
        params = {"page": page, "limit": limit}
        return await self._make_request("GET", f"/jobs/company/{company_id}", params=params)

    async def get_all_jobs(
        self,
        page: int = 1,
        limit: int = 20,
        active_only: bool = True,
    ) -> Dict[str, Any]:
        params = {"page": page, "limit": limit, "active": active_only}
        return await self._make_request("GET", "/jobs", params=params)

    async def get_all_companies(self, page: int = 1, query: Optional[str] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {"page": page}
        if query:
            params["q"] = query
        return await self._make_request("GET", "/companies", params=params)

    async def get_company_details(self, company_id: int) -> Dict[str, Any]:
        return await self._make_request("GET", f"/companies/{company_id}")

    async def get_backend_health(self) -> Dict[str, Any]:
        return await self._make_request("GET", "/")

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

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
        json_data = {
            "title": title,
            "description": description,
            "categoryId": category_id,
            "jobTypeId": job_type_id,
            "companyId": company_id,
            "salaryRange": salary_range,
            "requirements": requirements,
        }
        return await self._make_request("POST", "/jobs", json_data=json_data)


# Singleton instance
_backend_client: Optional[BackendAPIClient] = None


def get_backend_client() -> BackendAPIClient:
    """Get or create backend API client"""
    global _backend_client
    if _backend_client is None:
        _backend_client = BackendAPIClient()
    return _backend_client
