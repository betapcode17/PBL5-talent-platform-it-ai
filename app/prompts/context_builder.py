# app/prompts/context_builder.py
"""
Context Builder - Build structured, comprehensive context for LLM prompts
Following Prompt Engineering best practices
"""

from typing import Dict, List, Any, Optional
from datetime import datetime

# ============================================================================
# CONTEXT BUILDERS
# ============================================================================

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


def build_cv_analysis_context(
    cv_text: str,
    market_data: Optional[Dict[str, Any]] = None,
    user_profile: Optional[Dict[str, Any]] = None
) -> str:
    """Build context for CV analysis"""
    
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


def build_matching_context(
    cv_data: Dict[str, Any],
    job_data: Dict[str, Any],
    market_data: Optional[Dict[str, Any]] = None
) -> str:
    """Build context for CV-Job matching"""
    
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


def build_career_advice_context(
    user_profile: Dict[str, Any],
    market_data: Optional[Dict[str, Any]] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> str:
    """Build context for career advice"""
    
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


# ============================================================================
# HELPER FORMATTING FUNCTIONS
# ============================================================================

def format_job_for_context(job: Dict[str, Any], index: int = 0) -> str:
    """Format a single job for context display"""
    
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
    """Format multiple jobs for context"""
    
    formatted_jobs = []
    for i, job in enumerate(jobs[:max_jobs], 1):
        formatted_jobs.append(format_job_for_context(job, i))
    
    return "\n".join(formatted_jobs)


def format_conversation_for_context(
    messages: List[Dict[str, str]], 
    max_turns: int = 3
) -> str:
    """Format conversation history for context"""
    
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
    """Format skills list nicely"""
    return ", ".join(skills[:max_items])


def add_section(title: str, content: str) -> str:
    """Create a formatted section with title"""
    return f"\n=== {title} ===\n{content}\n"


# ============================================================================
# CONTEXT VALIDATION
# ============================================================================

def validate_context(context: str, min_length: int = 200) -> bool:
    """Validate context has minimum expected length"""
    return len(context) >= min_length


def check_data_presence(context: str) -> Dict[str, bool]:
    """Check what sections are present in context"""
    checks = {
        "has_user_profile": "USER PROFILE" in context or "User Profile" in context,
        "has_market_data": "MARKET" in context,
        "has_jobs": "JOB" in context or "RETRIEVED" in context,
        "has_history": "CONVERSATION" in context or "HISTORY" in context,
        "has_instructions": "INSTRUCTION" in context,
        "has_role": "ROLE" in context,
    }
    return checks
