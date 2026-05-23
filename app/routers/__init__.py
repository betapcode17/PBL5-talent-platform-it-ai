"""
API routers for the CV-Job Matcher.
Each router handles a group of related endpoints.
"""

from .cv import router as cv_router
from .matching import router as matching_router
from .candidates import router as candidates_router

__all__ = ['cv_router', 'matching_router', 'candidates_router']