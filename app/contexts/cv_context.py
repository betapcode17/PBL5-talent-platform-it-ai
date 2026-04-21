"""
CV Analysis Context Builder
Build context for CV analysis and improvement tasks
"""

from typing import Dict, List, Any, Optional


def build_cv_analysis_context(
    cv_text: str,
    market_data: Optional[Dict[str, Any]] = None,
    user_profile: Optional[Dict[str, Any]] = None
) -> str:
    """
    Build context for CV analysis
    
    Args:
        cv_text: The CV content to analyze
        market_data: Optional market data for reference
        user_profile: Optional user profile for context
    
    Returns:
        Formatted context string
    """
    
    context_parts = []
    
    context_parts.append("=== CV CONTENT ===")
    context_parts.append(cv_text[:1000] + "..." if len(cv_text) > 1000 else cv_text)
    context_parts.append("")
    
    if user_profile:
        context_parts.append("=== USER CONTEXT ===")
        context_parts.append(f"Target Level: {user_profile.get('target_level', '?')}")
        context_parts.append(f"Target Industries: {user_profile.get('industries', '?')}")
        context_parts.append(f"Goals: {user_profile.get('goals', '?')}")
        context_parts.append("")
    
    if market_data:
        context_parts.append("=== MARKET REFERENCE ===")
        top_skills = market_data.get('top_skills', [])
        if top_skills:
            context_parts.append(f"In-Demand Skills: {', '.join(s['name'] for s in top_skills[:5])}")
        context_parts.append("")
    
    context_parts.append("=== ANALYSIS INSTRUCTIONS ===")
    context_parts.append("1. Analyze CV structure and content quality")
    context_parts.append("2. Identify strengths (minimum 2)")
    context_parts.append("3. Identify areas for improvement (minimum 2)")
    context_parts.append("4. Provide specific, actionable suggestions")
    context_parts.append("5. Use constructive, encouraging tone")
    
    return "\n".join(context_parts)
