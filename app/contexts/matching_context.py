"""
Job-CV Matching Context Builder
Build context for matching CV with job requirements
"""

from typing import Dict, Any, Optional


def build_matching_context(
    cv_data: Dict[str, Any],
    job_data: Dict[str, Any],
    market_data: Optional[Dict[str, Any]] = None
) -> str:
    """
    Build context for CV-Job matching
    
    Args:
        cv_data: User CV data
        job_data: Job requirements data
        market_data: Optional market data for salary/trend context
    
    Returns:
        Formatted context string
    """
    
    context_parts = []
    
    context_parts.append("=== USER CV ===")
    context_parts.append(f"Level: {cv_data.get('level', '?')}")
    context_parts.append(f"Experience: {cv_data.get('experience', '?')} years")
    context_parts.append(f"Skills: {', '.join(cv_data.get('skills', []))}")
    context_parts.append(f"Background: {cv_data.get('background', '?')}")
    context_parts.append("")
    
    context_parts.append("=== JOB REQUIREMENTS ===")
    context_parts.append(f"Position: {job_data.get('title', '?')}")
    context_parts.append(f"Company: {job_data.get('company', '?')}")
    context_parts.append(f"Level: {job_data.get('level', '?')}")
    context_parts.append(f"Required Skills: {', '.join(job_data.get('required_skills', []))}")
    context_parts.append(f"Experience Required: {job_data.get('required_experience', '?')} years")
    context_parts.append(f"Salary: {job_data.get('salary', '?')}")
    context_parts.append("")
    
    if market_data:
        context_parts.append("=== MARKET CONTEXT ===")
        context_parts.append(f"Salary Range for {job_data.get('level')}: {market_data.get('salary_range', '?')}")
        context_parts.append(f"Demand: {market_data.get('demand_level', '?')}")
        context_parts.append("")
    
    context_parts.append("=== MATCHING METHODOLOGY ===")
    context_parts.append("1. Extract job requirements")
    context_parts.append("2. Map to user skills")
    context_parts.append("3. Calculate % overlap")
    context_parts.append("4. Identify skill gaps")
    context_parts.append("5. Assess experience level match")
    context_parts.append("6. Generate final score (0-100%)")
    context_parts.append("7. Recommend actions")
    
    return "\n".join(context_parts)
