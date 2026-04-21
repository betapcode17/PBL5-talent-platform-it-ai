"""
Job Search Context Builder
Build structured context for job search queries
"""

from typing import Dict, List, Any, Optional


def build_job_search_context(
    user_profile: Optional[Dict[str, Any]] = None,
    market_data: Optional[Dict[str, Any]] = None,
    retrieved_jobs: Optional[List[Dict[str, Any]]] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    user_message: Optional[str] = None
) -> str:
    """
    Build comprehensive context for job search queries
    Combines user profile, market data, retrieved jobs, and conversation
    """
    
    context_parts = []
    
    # 1. USER PROFILE SECTION
    if user_profile:
        context_parts.append("=== USER PROFILE ===")
        context_parts.append(f"Level: {user_profile.get('level', '?')}")
        context_parts.append(f"Experience: {user_profile.get('experience_years', '?')} years")
        
        skills = user_profile.get('skills', [])
        if skills:
            context_parts.append(f"Main Skills: {', '.join(skills[:5])}")
        
        context_parts.append(f"Industries: {user_profile.get('industries', 'Any')}")
        context_parts.append(f"Work Preference: {user_profile.get('work_type', 'Any')}")
        context_parts.append(f"Location: {user_profile.get('location', 'Any')}")
        context_parts.append(f"Salary Range: {user_profile.get('salary_min', '?')}-{user_profile.get('salary_max', '?')} million")
        context_parts.append("")
    
    # 2. MARKET DATA SECTION
    if market_data:
        context_parts.append("=== MARKET DATA ===")
        context_parts.append(f"Total Jobs Available: {market_data.get('total_jobs', '?')}")
        context_parts.append(f"Total Companies: {market_data.get('total_companies', '?')}")
        
        top_skills = market_data.get('top_skills', [])
        if top_skills:
            skills_str = ', '.join(
                f"{s['name']} ({s.get('count', 0)})" 
                for s in top_skills[:5]
            )
            context_parts.append(f"Top 5 Skills: {skills_str}")
        
        top_cats = market_data.get('top_categories', [])
        if top_cats:
            cats_str = ', '.join(
                f"{c['name']} ({c.get('count', 0)})" 
                for c in top_cats[:5]
            )
            context_parts.append(f"Top 5 Industries: {cats_str}")
        
        context_parts.append(f"Avg Salary ({user_profile.get('level', 'Unknown')}): {market_data.get('avg_salary', '?')} million")
        context_parts.append("")
    
    # 3. CONVERSATION HISTORY (Last 3 turns)
    if conversation_history and len(conversation_history) > 0:
        context_parts.append("=== RECENT CONVERSATION ===")
        for msg in conversation_history[-3:]:
            role = "🧑 You" if msg.get("role") == "user" else "🤖 Assistant"
            content = msg.get("content", "")[:100] + "..." if len(msg.get("content", "")) > 100 else msg.get("content", "")
            context_parts.append(f"{role}: {content}")
        context_parts.append("")
    
    # 4. CURRENT QUERY
    if user_message:
        context_parts.append("=== CURRENT QUERY ===")
        context_parts.append(f"User: {user_message}")
        context_parts.append("")
    
    # 5. RETRIEVED JOBS DATA
    if retrieved_jobs and len(retrieved_jobs) > 0:
        from .base_context import format_job_for_context
        context_parts.append(f"=== RETRIEVED JOBS ({len(retrieved_jobs)} items) ===")
        for i, job in enumerate(retrieved_jobs[:10], 1):
            job_str = format_job_for_context(job, i)
            context_parts.append(job_str)
        context_parts.append("")
    
    # 6. INSTRUCTIONS
    context_parts.append("=== INSTRUCTIONS ===")
    context_parts.append("1. Use ONLY data from RETRIEVED JOBS section")
    context_parts.append("2. DO NOT make up jobs or salary information")
    context_parts.append("3. RANK jobs by relevance to user profile")
    context_parts.append("4. EXPLAIN why each job matches")
    context_parts.append("5. Use bullet points and clear structure")
    context_parts.append("6. Respond in Vietnamese")
    
    return "\n".join(context_parts)
