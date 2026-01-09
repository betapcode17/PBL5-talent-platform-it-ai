"""
Prompts management for AI CV-Job Matcher.
Import prompts like: from app.prompts import qa_prompt, chart_insights_prompts
"""

from .rewrite_prompt import rewrite_prompt
from .qa_prompt import qa_prompt
from .cv_analysis_prompt import cv_analysis_prompt
from .cv_improvement_prompt import cv_improvement_prompt
from .chart_insights_prompt import chart_insights_prompts  

__all__ = [
    'rewrite_prompt',
    'qa_prompt',
    'cv_analysis_prompt',
    'cv_improvement_prompt',
    'chart_insights_prompts'
]