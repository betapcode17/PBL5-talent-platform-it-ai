"""
Prompts Package
Organized prompt management system with modular architecture:
- System Prompts (by intent: jobs, cv, matching, career)
- Context Builders (separate modules for each task type)
- Prompt Engineering (utilities for advanced techniques)
"""

# System Prompt Manager
from .system_prompt_manager import (
    CHAT_SYSTEM_PROMPTS,
    CHAIN_OF_THOUGHT_PROMPTS,
    QUALITY_CHECKLISTS,
    get_system_prompt,
    get_cot_prompt,
    get_quality_checklist,
    get_prompt_template,
    create_chat_template,
)

# Individual Prompt Modules (for direct access if needed)
from .default_system_prompt import get_default_prompt
from .jobs_system_prompt import (
    get_jobs_prompt,
    get_jobs_cot_prompt,
    get_jobs_quality_checklist,
)
from .cv_system_prompt import (
    get_cv_prompt,
    get_cv_quality_checklist,
)
from .matching_system_prompt import (
    get_matching_prompt,
    get_matching_cot_prompt,
    get_matching_quality_checklist,
)
from .career_system_prompt import (
    get_career_prompt,
)

# Legacy imports for backward compatibility
try:
    from .chart_insights_prompt import chart_insights_prompts
    from .cv_analysis_prompt import cv_analysis_prompt
    from .cv_improvement_prompt import cv_improvement_prompt
    from .qa_prompt import qa_prompt
    from .retrieval_prompt import retrieval_prompt
    from .rewrite_prompt import rewrite_prompt
    from .templates import *
except ImportError:
    pass

__all__ = [
    # System Prompt Manager (Core)
    'CHAT_SYSTEM_PROMPTS',
    'CHAIN_OF_THOUGHT_PROMPTS',
    'QUALITY_CHECKLISTS',
    'get_system_prompt',
    'get_cot_prompt',
    'get_quality_checklist',
    'get_prompt_template',
    'create_chat_template',
    
    # Individual Prompts (Direct access)
    'get_default_prompt',
    'get_jobs_prompt',
    'get_jobs_cot_prompt',
    'get_jobs_quality_checklist',
    'get_cv_prompt',
    'get_cv_quality_checklist',
    'get_matching_prompt',
    'get_matching_cot_prompt',
    'get_matching_quality_checklist',
    'get_career_prompt',
    
    # Legacy (Backward compatibility)
    'rewrite_prompt',
    'qa_prompt',
    'cv_analysis_prompt',
    'cv_improvement_prompt',
    'chart_insights_prompts',
    'retrieval_prompt',
]