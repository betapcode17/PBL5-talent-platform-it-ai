"""
Context Builders Package
Organized context builders for different intents/tasks
"""

from .base_context import (
    format_job_for_context,
    format_jobs_for_context,
    format_conversation_for_context,
    format_skills_list,
    add_section,
    format_user_profile,
    format_market_data,
    validate_context,
    check_data_presence,
    validate_jobs_data,
    count_context_sections,
)

from .jobs_context import build_job_search_context
from .cv_context import build_cv_analysis_context
from .matching_context import build_matching_context
from .career_context import build_career_advice_context

__all__ = [
    # Base utilities
    "format_job_for_context",
    "format_jobs_for_context",
    "format_conversation_for_context",
    "format_skills_list",
    "add_section",
    "format_user_profile",
    "format_market_data",
    "validate_context",
    "check_data_presence",
    "validate_jobs_data",
    "count_context_sections",
    
    # Context builders
    "build_job_search_context",
    "build_cv_analysis_context",
    "build_matching_context",
    "build_career_advice_context",
]
