"""
API routers for the CV-Job Matcher.
Each router handles a group of related endpoints.
"""

from .cv import router as cv_router
from .jobs import router as jobs_router
from .matching import router as matching_router
from .utils import router as utils_router

__all__ = ['cv_router', 'jobs_router', 'matching_router', 'utils_router']