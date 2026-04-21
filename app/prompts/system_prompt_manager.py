"""
System Prompt Manager
Centralized point to get system prompts for different intents
"""

from langchain_core.prompts import ChatPromptTemplate
from .default_system_prompt import get_default_prompt
from .jobs_system_prompt import get_jobs_prompt, get_jobs_cot_prompt, get_jobs_quality_checklist
from .cv_system_prompt import get_cv_prompt, get_cv_quality_checklist
from .matching_system_prompt import get_matching_prompt, get_matching_cot_prompt, get_matching_quality_checklist
from .career_system_prompt import get_career_prompt

# ============================================================================
# SYSTEM PROMPTS REGISTRY
# ============================================================================

CHAT_SYSTEM_PROMPTS = {
    "default": get_default_prompt,
    "jobs": get_jobs_prompt,
    "cv": get_cv_prompt,
    "matching": get_matching_prompt,
    "career": get_career_prompt,
}

# Chain-of-Thought prompts
CHAIN_OF_THOUGHT_PROMPTS = {
    "matching": get_matching_cot_prompt,
    "jobs": get_jobs_cot_prompt,
}

# Quality checklists
QUALITY_CHECKLISTS = {
    "jobs": get_jobs_quality_checklist,
    "cv": get_cv_quality_checklist,
    "matching": get_matching_quality_checklist,
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_system_prompt(context_type: str = "jobs") -> str:
    """
    Get enhanced system prompt for context type
    
    Args:
        context_type: Type of context (default, jobs, cv, matching, career)
    
    Returns:
        System prompt string
    """
    prompt_getter = CHAT_SYSTEM_PROMPTS.get(context_type)
    if prompt_getter:
        return prompt_getter()
    return CHAT_SYSTEM_PROMPTS["default"]()


def get_cot_prompt(task_type: str) -> str:
    """
    Get chain-of-thought prompt for task type
    
    Args:
        task_type: Type of task (matching, jobs)
    
    Returns:
        Chain-of-thought instruction string
    """
    prompt_getter = CHAIN_OF_THOUGHT_PROMPTS.get(task_type)
    if prompt_getter:
        return prompt_getter()
    return ""


def get_quality_checklist(context_type: str = "jobs") -> list:
    """
    Get quality checklist for context type
    
    Args:
        context_type: Type of context (jobs, cv, matching)
    
    Returns:
        Quality checklist list
    """
    checklist_getter = QUALITY_CHECKLISTS.get(context_type)
    if checklist_getter:
        return checklist_getter()
    return []


# ============================================================================
# PROMPT TEMPLATES (for LangChain)
# ============================================================================

def create_chat_template(context_type: str = "default") -> ChatPromptTemplate:
    """Create a LangChain ChatPromptTemplate for a context type"""
    system_prompt = get_system_prompt(context_type)
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])


# Prebuilt templates
PROMPT_TEMPLATES = {
    "default": create_chat_template("default"),
    "jobs": create_chat_template("jobs"),
    "cv": create_chat_template("cv"),
    "matching": create_chat_template("matching"),
    "career": create_chat_template("career"),
}


def get_prompt_template(context_type: str = "jobs") -> ChatPromptTemplate:
    """Get prompt template for a context type"""
    return PROMPT_TEMPLATES.get(context_type, PROMPT_TEMPLATES["default"])
