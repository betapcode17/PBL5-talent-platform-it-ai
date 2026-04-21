"""
Career Advice Context Builder
Build context for career advising conversations
"""

from typing import Dict, List, Any, Optional


def build_career_advice_context(
    user_profile: Dict[str, Any],
    market_data: Optional[Dict[str, Any]] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Build context for career advice
    
    Args:
        user_profile: User career profile and goals
        market_data: Optional market trends and data
        conversation_history: Optional conversation history
    
    Returns:
        Formatted context string
    """
    
    context_parts = []
    
    context_parts.append("=== USER PROFILE ===")
    context_parts.append(f"Current Level: {user_profile.get('current_level', '?')}")
    context_parts.append(f"Experience: {user_profile.get('experience', '?')} years")
    context_parts.append(f"Current Role: {user_profile.get('current_role', '?')}")
    context_parts.append(f"Skills: {', '.join(user_profile.get('skills', []))}")
    context_parts.append(f"Goals: {user_profile.get('goals', '?')}")
    context_parts.append(f"Challenges: {user_profile.get('challenges', '?')}")
    context_parts.append("")
    
    if market_data:
        context_parts.append("=== MARKET TRENDS ===")
        context_parts.append(f"Growing Roles: {market_data.get('growing_roles', '?')}")
        context_parts.append(f"In-Demand Skills: {', '.join(market_data.get('top_skills', [])[:5])}")
        context_parts.append(f"Salary Progression: {market_data.get('salary_progression', '?')}")
        context_parts.append("")
    
    if conversation_history:
        context_parts.append("=== CONVERSATION HISTORY ===")
        for msg in conversation_history[-2:]:
            role = "User" if msg.get("role") == "user" else "Advisor"
            context_parts.append(f"{role}: {msg.get('content', '')[:100]}...")
        context_parts.append("")
    
    context_parts.append("=== ADVICE GUIDELINES ===")
    context_parts.append("1. Understand user's current situation deeply")
    context_parts.append("2. Analyze market opportunities")
    context_parts.append("3. Identify skill gaps and learning opportunities")
    context_parts.append("4. Create actionable 12-month roadmap")
    context_parts.append("5. Provide realistic, encouraging guidance")
    context_parts.append("6. Consider work-life balance")
    
    return "\n".join(context_parts)
