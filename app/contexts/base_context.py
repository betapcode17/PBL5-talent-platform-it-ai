"""
Base Context Utilities
Common formatting and validation functions for context building
"""

from typing import Dict, List, Any, Optional


# ============================================================================
# HELPER FORMATTING FUNCTIONS
# ============================================================================

def format_job_for_context(job: Dict[str, Any], index: int = 0) -> str:
    """
    Format a single job for context display
    
    Args:
        job: Job data dictionary
        index: Optional index number
    
    Returns:
        Formatted job string
    """
    
    parts = []
    
    if index > 0:
        parts.append(f"{index}. **{job.get('title', 'Unknown')}** @ {job.get('company', 'Unknown')}")
    else:
        parts.append(f"**{job.get('title', 'Unknown')}** @ {job.get('company', 'Unknown')}")
    
    parts.append(f"   - Location: {job.get('location', 'N/A')}")
    parts.append(f"   - Salary: {job.get('salary', 'N/A')}")
    parts.append(f"   - Experience: {job.get('experience', 'N/A')}")
    
    skills = job.get('skills', [])
    if skills:
        parts.append(f"   - Skills: {', '.join(skills[:3])}")
    
    url = job.get('url')
    if url:
        parts.append(f"   - URL: {url}")
    
    return "\n".join(parts)


def format_jobs_for_context(jobs: List[Dict[str, Any]], max_jobs: int = 10) -> str:
    """
    Format multiple jobs for context
    
    Args:
        jobs: List of job dictionaries
        max_jobs: Maximum number of jobs to include
    
    Returns:
        Formatted jobs string
    """
    
    formatted_jobs = []
    for i, job in enumerate(jobs[:max_jobs], 1):
        formatted_jobs.append(format_job_for_context(job, i))
    
    return "\n".join(formatted_jobs)


def format_conversation_for_context(
    messages: List[Dict[str, str]], 
    max_turns: int = 3
) -> str:
    """
    Format conversation history for context
    
    Args:
        messages: List of message dictionaries with role/content
        max_turns: Maximum conversation turns to include
    
    Returns:
        Formatted conversation string
    """
    
    parts = []
    for msg in messages[-max_turns*2:]:
        role = "🧑 You" if msg.get("role") == "user" else "🤖 Assistant"
        content = msg.get("content", "")
        # Truncate if too long
        if len(content) > 150:
            content = content[:150] + "..."
        parts.append(f"{role}: {content}")
    
    return "\n".join(parts)


def format_skills_list(skills: List[str], max_items: int = 10) -> str:
    """
    Format skills list nicely
    
    Args:
        skills: List of skill strings
        max_items: Maximum items to display
    
    Returns:
        Formatted skills string
    """
    return ", ".join(skills[:max_items])


def add_section(title: str, content: str) -> str:
    """
    Create a formatted section with title
    
    Args:
        title: Section title
        content: Section content
    
    Returns:
        Formatted section string
    """
    return f"\n=== {title} ===\n{content}\n"


def format_user_profile(profile: Dict[str, Any]) -> str:
    """
    Format user profile for display
    
    Args:
        profile: User profile dictionary
    
    Returns:
        Formatted profile string
    """
    parts = []
    parts.append(f"Level: {profile.get('level', '?')}")
    parts.append(f"Experience: {profile.get('experience_years', '?')} years")
    parts.append(f"Location: {profile.get('location', 'Any')}")
    
    skills = profile.get('skills', [])
    if skills:
        parts.append(f"Skills: {', '.join(skills[:5])}")
    
    return "\n".join(parts)


def format_market_data(market: Dict[str, Any]) -> str:
    """
    Format market data for display
    
    Args:
        market: Market data dictionary
    
    Returns:
        Formatted market data string
    """
    parts = []
    parts.append(f"Total Jobs: {market.get('total_jobs', '?')}")
    parts.append(f"Total Companies: {market.get('total_companies', '?')}")
    
    top_skills = market.get('top_skills', [])
    if top_skills:
        skills_list = ', '.join(s['name'] for s in top_skills[:5])
        parts.append(f"Top Skills: {skills_list}")
    
    parts.append(f"Avg Salary: {market.get('avg_salary', '?')}")
    
    return "\n".join(parts)


# ============================================================================
# CONTEXT VALIDATION
# ============================================================================

def validate_context(context: str, min_length: int = 200) -> bool:
    """
    Validate context has minimum expected length
    
    Args:
        context: The context string to validate
        min_length: Minimum required length
    
    Returns:
        True if valid, False otherwise
    """
    return len(context) >= min_length


def check_data_presence(context: str) -> Dict[str, bool]:
    """
    Check what sections are present in context
    
    Args:
        context: The context string to check
    
    Returns:
        Dictionary with presence checks
    """
    checks = {
        "has_user_profile": "USER PROFILE" in context or "User Profile" in context,
        "has_market_data": "MARKET" in context,
        "has_jobs": "JOB" in context or "RETRIEVED" in context,
        "has_history": "CONVERSATION" in context or "HISTORY" in context,
        "has_instructions": "INSTRUCTION" in context,
        "has_role": "ROLE" in context,
    }
    return checks


def validate_jobs_data(jobs: List[Dict[str, Any]]) -> Dict[str, bool]:
    """
    Validate jobs data structure
    
    Args:
        jobs: List of job dictionaries
    
    Returns:
        Dictionary with validation results
    """
    if not jobs:
        return {"is_empty": True, "is_valid": False}
    
    sample_job = jobs[0]
    checks = {
        "has_title": "title" in sample_job,
        "has_company": "company" in sample_job,
        "has_skills": "skills" in sample_job,
        "has_salary": "salary" in sample_job,
        "has_url": "url" in sample_job,
    }
    checks["is_valid"] = all(checks.values())
    checks["is_empty"] = False
    
    return checks


def count_context_sections(context: str) -> int:
    """
    Count number of sections in context
    
    Args:
        context: The context string
    
    Returns:
        Number of sections found
    """
    return context.count("===")
