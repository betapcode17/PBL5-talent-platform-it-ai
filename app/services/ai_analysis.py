"""
AI Analysis Functions - Phân tích CV và Jobs bằng Local Ollama hoặc Gemini
"""
import logging
import json
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Any, Tuple

from app.services.llm_service import get_llm_service
from app.services.chroma_utils import get_vectorstore
from typing import cast

# Get LLM service (Ollama or Gemini based on config)
def get_llm():
    """Get LLM instance (Ollama or Gemini)"""
    return get_llm_service()

async def analyze_cv_insights(cv_info: Dict) -> Dict[str, Any]:
    """
    Phân tích CV chuyên sâu - Đánh giá chất lượng, điểm mạnh/yếu
    
    Args:
        cv_info: Thông tin CV đã parse
        
    Returns:
        Dict chứa quality_score, strengths, weaknesses, completeness, market_fit
    """
    try:
        # Get LLM instance (works with both Ollama and Gemini)
        llm_instance = get_llm()
        from app.prompts import cv_analysis_prompt
        
        # Format input for prompt
        formatted_input = {
            "name": cv_info.get('name', 'N/A'),
            "email": cv_info.get('email', 'N/A'),
            "phone": cv_info.get('phone', 'N/A'),
            "skills": ', '.join(cv_info.get('skills', [])),
            "career_objective": cv_info.get('career_objective', 'N/A'),
            "experience_count": len(cv_info.get('experience', [])),
            "education_count": len(cv_info.get('education', []))
        }
        
        # Invoke prompt - compatible with both Ollama and Gemini
        prompt_text = cv_analysis_prompt.format(**formatted_input)
        
        # Try to use ainvoke if available (Gemini), otherwise use generate_response
        if hasattr(llm_instance.llm, 'ainvoke'):
            response = await llm_instance.llm.ainvoke(prompt_text)
            content = response.content.strip() # type: ignore
        else:
            # Ollama path - use async wrapper
            response = await llm_instance.llm.ainvoke(prompt_text)
            content = response.content.strip() # type: ignore
        
        # Remove markdown code blocks if present
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        result = json.loads(content)

        # Validate và fix completeness_score (KHÔNG BAO GIỜ ÂM)
        if 'completeness_score' in result:
            result['completeness_score'] = max(0.0, min(1.0, float(result['completeness_score'])))

        # Validate quality_score
        if 'quality_score' in result:
            result['quality_score'] = max(0.0, min(10.0, float(result['quality_score'])))

        # Validate market_fit_score
        if 'market_fit_score' in result:
            result['market_fit_score'] = max(0.0, min(1.0, float(result['market_fit_score'])))

        logging.info(f"✓ Phân tích CV thành công: quality_score={result.get('quality_score')}, completeness={result.get('completeness_score')}")
        # Ensure required top-level keys exist and have expected types (do not hallucinate values)
        normalized = {}
        normalized['summary'] = str(result.get('summary') or '')
        
        # Validate strengths: ensure all have evidence (point 4, 20)
        strengths = result.get('strengths') if isinstance(result.get('strengths'), list) else []
        validated_strengths = []
        for s in strengths:
            if isinstance(s, dict) and s.get('evidence') and s.get('point'):  # Only include if has evidence
                validated_strengths.append(s)
        normalized['strengths'] = validated_strengths
        
        # Validate weaknesses: ensure all have evidence (point 4, 20)
        weaknesses = result.get('weaknesses') if isinstance(result.get('weaknesses'), list) else []
        validated_weaknesses = []
        for w in weaknesses:
            if isinstance(w, dict) and w.get('evidence') and w.get('issue'):  # Only include if has evidence
                validated_weaknesses.append(w)
        normalized['weaknesses'] = validated_weaknesses
        
        normalized['improvement_suggestions'] = result.get('improvement_suggestions') if isinstance(result.get('improvement_suggestions'), list) else []

        # ATS block
        ats = result.get('ats') or {}
        normalized['ats'] = {
            'score': int(ats.get('score') or result.get('quality_score') or 0),
            'readability': ats.get('readability') or 'unknown',
            'keyword_quality': ats.get('keyword_quality') or 'unknown',
            'technical_keyword_strength': ats.get('technical_keyword_strength') or 'unknown',
            'recruiter_impression': ats.get('recruiter_impression') or '',
            'formatting_quality': ats.get('formatting_quality') or 'unknown'
        }

        # career_fit
        normalized['career_fit'] = result.get('career_fit') if isinstance(result.get('career_fit'), list) else []

        # technical_analysis with validation rules (points 7, 8, 15, 16)
        ta = result.get('technical_analysis') or {}
        
        # Point 7: Don't return "Unknown" if CV has clear technical projects and skills
        current_level = ta.get('current_level') or 'Unknown'
        skills = cv_info.get('skills', [])
        has_backend_stack = any(s.lower() in ['java', 'spring', 'laravel', 'nodejs', 'python', 'flask', 'django', 'sql', 'rest api', 'docker', 'git'] for s in skills)
        has_projects = len(cv_info.get('experience', [])) > 0 or cv_info.get('has_projects', False)
        if current_level == 'Unknown' and has_backend_stack and has_projects:
            current_level = 'Junior'  # Infer Junior if has backend stack + projects
        
        # Point 8: Quality score should be reasonable (student intern CV not > 8.5)
        quality_score = float(result.get('quality_score') or 5.0)
        if current_level == 'Intern' and quality_score > 8.5:
            quality_score = 8.5  # Cap intern quality score at 8.5
        
        # Point 16: Ensure missing_technical_skills includes testing, CI/CD, cloud, deployment if not present
        missing_skills = ta.get('missing_technical_skills') if isinstance(ta.get('missing_technical_skills'), list) else []
        # Check if CV mentions testing, CI/CD, cloud, deployment
        cv_text_lower = ' '.join(str(cv_info.get(k, '')) for k in cv_info.keys()).lower()
        if 'test' not in cv_text_lower and 'Unit Testing' not in missing_skills: # type: ignore
            missing_skills.append('Unit Testing') # type: ignore
        if 'ci/cd' not in cv_text_lower and 'github action' not in cv_text_lower and 'CI/CD' not in missing_skills:    # type: ignore
            missing_skills.append('CI/CD') # type: ignore
        if 'deployment' not in cv_text_lower and 'Deployment' not in missing_skills: # type: ignore
            missing_skills.append('Deployment') # type: ignore
        
        normalized['technical_analysis'] = {
            'current_level': current_level,
            'backend_readiness': ta.get('backend_readiness') or 'unknown',
            'strongest_stack': ta.get('strongest_stack') if isinstance(ta.get('strongest_stack'), list) else [],
            'strongest_concepts': ta.get('strongest_concepts') if isinstance(ta.get('strongest_concepts'), list) else [],
            'missing_technical_skills': missing_skills,
            'project_complexity': ta.get('project_complexity') or 'unknown',
            'architecture_understanding': ta.get('architecture_understanding') or 'unknown',
            'production_readiness': ta.get('production_readiness') or 'unknown'
        }

        # Validate learning_roadmap: no placeholders, specific projects/resources (points 11, 12)
        roadmap = result.get('learning_roadmap') if isinstance(result.get('learning_roadmap'), list) else []
        validated_roadmap = []
        placeholder_keywords = ['chưa', 'something', 'N/A', 'TBD', 'cần', 'TODO']
        for step in roadmap:
            if isinstance(step, dict):
                # Skip if contains placeholders
                all_fields = str(step).lower()
                if any(kw in all_fields for kw in placeholder_keywords):
                    continue
                # Skip if empty objective/project/milestone
                if not step.get('objective') or not step.get('project') or not step.get('milestone'):
                    continue
                validated_roadmap.append(step)
        normalized['learning_roadmap'] = validated_roadmap
        
        normalized['market_competitiveness'] = result.get('market_competitiveness') if isinstance(result.get('market_competitiveness'), dict) else {}
        
        # Points 1, 2: Fix has_projects and missing_sections
        missing_sections = result.get('missing_sections') if isinstance(result.get('missing_sections'), list) else []
        # Point 1: Don't report missing "projects" if CV has PROJECTS section
        has_projects_section = cv_info.get('has_projects', False)
        if 'projects' in [s.lower() for s in missing_sections] and has_projects_section:
            missing_sections = [s for s in missing_sections if s.lower() != 'projects']
        normalized['missing_sections'] = missing_sections
        
        # Preserve some legacy scores if present
        if 'quality_score' in result:
            normalized['quality_score'] = max(0.0, min(10.0, quality_score))
        if 'completeness_score' in result:
            normalized['completeness_score'] = result.get('completeness_score')

        return normalized
        
    except json.JSONDecodeError as e:
        logging.error(f"✗ Lỗi parse JSON: {e}")
        logging.error(f"Response content: {content}")
        # Return default values
        return {
            "quality_score": 5.0,
            "completeness_score": 0.5,
            "has_portfolio": False,
            "has_certifications": False,
            "has_projects": False,
            "missing_sections": ["Unknown"],
            "market_fit_score": 0.5,
            "experience_level": "Unknown",
            "salary_range": "N/A",
            "competitive_score": 5.0,
            "strengths": ["Cần phân tích thêm"],
            "weaknesses": ["Cần phân tích thêm"]
        }
    except Exception as e:
        logging.error(f"✗ Lỗi phân tích CV: {e}")
        raise


async def generate_cv_improvements(cv_info: Dict, insights: Dict) -> List[Dict[str, Any]]:
    """
    Tạo gợi ý cải thiện CV cụ thể
    
    Args:
        cv_info: Thông tin CV
        insights: Kết quả phân tích từ analyze_cv_insights
        
    Returns:
        List các gợi ý cải thiện
    """
    try:
        # Get LLM instance (works with both Ollama and Gemini)
        llm_instance = get_llm()
        from app.prompts import cv_improvement_prompt
        
        # Format input for prompt
        # Normalize weaknesses: support list[str] or list[dict]
        raw_weaknesses = insights.get('weaknesses', []) if insights else []
        weakness_texts = []
        if isinstance(raw_weaknesses, list):
            for w in raw_weaknesses:
                if isinstance(w, dict):
                    # prefer 'issue' field
                    issue = w.get('issue') or w.get('point') or str(w)
                    weakness_texts.append(str(issue))
                else:
                    weakness_texts.append(str(w))

        formatted_input = {
            "skills": ', '.join(cv_info.get('skills', [])) if cv_info.get('skills') else 'Chưa có',
            "experience_count": len(cv_info.get('experience', [])),
            "education_count": len(cv_info.get('education', [])),
            "career_objective": cv_info.get('career_objective', 'Chưa có')[:200] + '...' if cv_info.get('career_objective') else 'Chưa có',
            "weaknesses": ', '.join(weakness_texts),
            "missing_sections": ', '.join(insights.get('missing_sections', [])) if insights else '',
            "has_portfolio": insights.get('has_portfolio', False) if insights else False,
            "has_certifications": insights.get('has_certifications', False) if insights else False,
            "has_projects": insights.get('has_projects', False) if insights else False
        }
        
        # Invoke prompt
        prompt_text = cv_improvement_prompt.format(**formatted_input)
        
        # Try to use ainvoke if available (Gemini), otherwise use generate_response
        if hasattr(llm_instance.llm, 'ainvoke'):
            response = await llm_instance.llm.ainvoke(prompt_text)
            content = response.content.strip() # type: ignore
        else:
            # Ollama path - use async wrapper
            response = await llm_instance.llm.ainvoke(prompt_text)
            content = response.content.strip() # type: ignore

        # Remove markdown code blocks
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        improvements = json.loads(content)

        # Validate và fix data types
        for imp in improvements:
            # Convert current to list if it's a string
            if isinstance(imp.get('current'), str):
                imp['current'] = [imp['current']] if imp['current'] else None

            # Convert suggested_add to list if it's a string
            if isinstance(imp.get('suggested_add'), str):
                imp['suggested_add'] = [imp['suggested_add']] if imp['suggested_add'] else None

        logging.info(f"✓ Tạo {len(improvements)} gợi ý cải thiện")
        return improvements
        
    except json.JSONDecodeError as e:
        logging.error(f"✗ Lỗi parse JSON: {e}")
        logging.error(f"Response: {content}")
        return [
            {
                "section": "general",
                "current": None,
                "suggested_add": None,
                "suggestion": "Cần phân tích thêm để đưa ra gợi ý cụ thể",
                "reason": "Lỗi phân tích",
                "priority": "medium",
                "impact": "N/A"
            }
        ]
    except Exception as e:
        logging.error(f"✗ Lỗi tạo gợi ý: {e}")
        raise


async def generate_learning_roadmap(cv_info: Dict, insights: Dict, job_analysis: Dict) -> List[Dict[str, Any]]:
    """
    Generate a time-boxed learning roadmap based on CV, insights, and job analysis.
    Falls back to a heuristic roadmap if LLM is unavailable or fails.
    """
    try:
        llm_instance = get_llm()
        from app.prompts import learning_roadmap_prompt

        skills = ', '.join(cv_info.get('skills', [])) if cv_info.get('skills') else 'Chưa có'
        # weaknesses may be list[str] or list[dict]
        wk_raw = insights.get('weaknesses', []) if insights else []
        if isinstance(wk_raw, list):
            wk_items = []
            for w in wk_raw:
                if isinstance(w, dict):
                    wk_items.append(w.get('issue') or w.get('point') or str(w))
                else:
                    wk_items.append(str(w))
            weaknesses = ', '.join(wk_items)
        else:
            weaknesses = str(wk_raw or '')
        # learning_suggestions may be a list of dicts with 'skill' and 'priority'
        suggs = job_analysis.get('learning_suggestions', []) if job_analysis else []
        sugg_str = ', '.join([s.get('skill') for s in suggs]) if suggs else ''

        formatted = {
            'skills': skills,
            'weaknesses': weaknesses,
            'learning_suggestions': sugg_str
        }

        prompt_text = learning_roadmap_prompt.format(**formatted) # type: ignore

        # Attempt LLM call (async)
        if hasattr(llm_instance.llm, 'ainvoke'):
            response = await llm_instance.llm.ainvoke(prompt_text)
            content = response.content.strip()  # type: ignore
        else:
            response = await llm_instance.llm.ainvoke(prompt_text)
            content = response.content.strip()  # type: ignore

        # strip code fences
        content = re.sub(r"^```json|```$|^```", "", content).strip()
        roadmap = json.loads(content)
        # Basic validation: ensure list of dicts
        if not isinstance(roadmap, list):
            raise ValueError("Roadmap must be a list")
        return roadmap
    except Exception as e:
        logging.warning(f"Learning roadmap generation failed ({type(e).__name__}): {e}. Using heuristic fallback.")
        # Heuristic fallback: build roadmap from top missing skills in job_analysis
        fallback: List[Dict[str, Any]] = []
        top_skills = [s.get('skill') for s in (job_analysis.get('learning_suggestions') or [])][:5]
        # If none, use weaknesses keywords
        if not top_skills and insights:
            # extract noun-like tokens from weaknesses
            wk = insights.get('weaknesses', [])
            for w in wk:
                # try to grab words with length>3
                words = re.findall(r"[A-Za-zĐđÀ-ỹ]{3,}", str(w))
                if words:
                    top_skills.append(words[0])

        # Build simple phases
        if top_skills:
            # quick start: 2-4 weeks fundamentals
            fallback.append({
                "phase": "0-4 weeks",
                "duration_weeks": 4,
                "objectives": [f"Nắm bắt cơ bản: {', '.join(top_skills[:2])}"] ,
                "resources": [{"name": "Official docs / free tutorial", "url": "https://www.google.com/search?q=learn+%s" % top_skills[0]}],
                "projects": [f"Mini project: build small app using {top_skills[0]}"] ,
                "milestones": ["Hoàn thành 3 tutorial cơ bản"]
            })
            # intermediate: 1-3 months
            fallback.append({
                "phase": "1-3 months",
                "duration_weeks": 8,
                "objectives": [f"Áp dụng {top_skills[0]} vào project hoàn chỉnh", f"Học thêm: {', '.join(top_skills[1:3])}"] ,
                "resources": [{"name": "Coursera/YouTube course", "url": "https://www.coursera.org/"}],
                "projects": [f"Build and deploy an end-to-end app using {top_skills[0]}"] ,
                "milestones": ["Deploy working app; write README and tests"]
            })
            # advanced
            fallback.append({
                "phase": "3-6 months",
                "duration_weeks": 12,
                "objectives": ["Deepen architecture and best practices", "Contribute to open-source or publish project"],
                "resources": [{"name": "Advanced course / Books", "url": "https://www.oreilly.com/"}],
                "projects": ["Refactor project for scalability; add CI/CD"],
                "milestones": ["Project accepted or used by others / portfolio updated"]
            })
        else:
            # General fallback roadmap
            fallback = [
                {
                    "phase": "0-4 weeks",
                    "duration_weeks": 4,
                    "objectives": ["Tổ chức CV rõ ràng: sections, contact, skills list"],
                    "resources": [{"name": "CV checklist", "url": "https://www.google.com/search?q=cv+checklist"}],
                    "projects": ["Viết phần Projects 1-2 items với links"],
                    "milestones": ["CV có đầy đủ sections"]
                }
            ]

        return fallback


def generate_why_match(cv_skills: List[str], job_skills: List[str], job_title: str) -> str:
    """
    Tạo lý do tại sao CV phù hợp với job
    
    Args:
        cv_skills: Kỹ năng từ CV
        job_skills: Kỹ năng yêu cầu của job
        job_title: Tiêu đề job
        
    Returns:
        Chuỗi mô tả lý do phù hợp
    """
    matched_skills = set(cv_skills) & set(job_skills)
    
    if not matched_skills:
        return f"Có thể phù hợp với vị trí {job_title}"
    
    skills_str = ", ".join(list(matched_skills)[:3])
    return f"Phù hợp với kỹ năng: {skills_str}"


def _normalize_skill_entries(value: Any) -> List[str]:
    """Normalize skill-like text into lower-case phrases for comparison."""
    if not value:
        return []

    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = re.split(r"[;,/|\n]+", str(value))

    aliases = {
        "ai": "artificial intelligence",
        "ml": "machine learning",
        "dl": "deep learning",
        "nlp": "natural language processing",
        "cv": "computer vision",
        "js": "javascript",
        "ts": "typescript",
        "nodejs": "nodejs",
        "reactjs": "react",
        "postgres": "postgresql",
    }

    normalized: List[str] = []
    for item in raw_items:
        text = re.sub(r"\s+", " ", str(item).strip().lower())
        text = text.strip(" .,:;()[]{}")
        if not text:
            continue
        text = aliases.get(text, text)
        normalized.append(text)
    return normalized


def _build_cv_job_query(cv_info: Dict[str, Any]) -> str:
    parts: List[str] = []
    name = str(cv_info.get("name") or "").strip()
    objective = str(cv_info.get("career_objective") or "").strip()
    skills = " ".join(_normalize_skill_entries(cv_info.get("skills", [])))
    parts.extend([name, objective, skills])

    for exp in cv_info.get("experience", []) or []:
        if isinstance(exp, dict):
            parts.extend([
                str(exp.get("title") or "").strip(),
                str(exp.get("company") or "").strip(),
                str(exp.get("description") or "").strip(),
            ])

    for edu in cv_info.get("education", []) or []:
        if isinstance(edu, dict):
            parts.extend([
                str(edu.get("major") or "").strip(),
                str(edu.get("degree") or "").strip(),
                str(edu.get("school") or "").strip(),
            ])

    query = " ".join(part for part in parts if part)
    if not query:
        query = " ".join(_normalize_skill_entries(cv_info.get("skills", []))) or objective or name or "CV"
    return query


async def analyze_cv_against_jobs(cv_info: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
    """Analyze a CV against job data stored in ChromaDB.

    Returns matched jobs plus skill-gap recommendations derived from top job results.
    """
    vectorstore = get_vectorstore("jobs")
    query = _build_cv_job_query(cv_info)
    cv_skills = set(_normalize_skill_entries(cv_info.get("skills", [])))

    try:
        raw_results: List[Tuple[Any, float]] = vectorstore.similarity_search_with_relevance_scores(
            query,
            k=max(top_k * 3, top_k),
        )
    except Exception:
        raw_results = []

    if not raw_results:
        try:
            docs = vectorstore.similarity_search(query, k=max(top_k * 3, top_k))
            raw_results = [(doc, 0.5) for doc in docs]
        except Exception:
            raw_results = []

    skill_counter: Counter[str] = Counter()
    skill_examples: Dict[str, List[str]] = defaultdict(list)
    matched_jobs: List[Dict[str, Any]] = []

    for doc, relevance in raw_results:
        metadata = getattr(doc, "metadata", {}) or {}
        job_title = str(metadata.get("job_title") or metadata.get("name") or "Unknown job")
        company_name = str(metadata.get("company") or metadata.get("name") or metadata.get("company_name") or "Unknown Company")
        job_skills = _normalize_skill_entries(metadata.get("skills"))
        if not job_skills:
            job_skills = _normalize_skill_entries(metadata.get("candidate_requirements"))
        if not job_skills:
            job_skills = _normalize_skill_entries(metadata.get("job_description"))

        job_skill_set = set(job_skills)
        matched_skills = sorted(cv_skills & job_skill_set)
        missing_skills = sorted(job_skill_set - cv_skills)
        overlap_ratio = len(matched_skills) / max(1, len(job_skill_set)) if job_skill_set else 0.0
        match_score = max(0.0, min(1.0, float(relevance) * 0.7 + overlap_ratio * 0.3))

        for skill in missing_skills:
            skill_counter[skill] += 1
            if job_title not in skill_examples[skill]:
                skill_examples[skill].append(job_title)

        matched_jobs.append({
            "job_id": int(metadata.get("job_id")) if str(metadata.get("job_id") or "").isdigit() else None, # type: ignore
            "job_title": job_title,
            "company_name": company_name,
            "match_score": round(match_score, 6),
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "why_match": generate_why_match(list(cv_skills), job_skills, job_title),
            "salary": str(metadata.get("salary") or "N/A"),
            "work_location": str(metadata.get("location") or metadata.get("work_location") or "N/A"),
            "work_type": str(metadata.get("work_type") or "N/A"),
        })

    matched_jobs.sort(key=lambda item: item["match_score"], reverse=True)
    matched_jobs = matched_jobs[:top_k]

    learning_suggestions: List[Dict[str, Any]] = []
    for skill, count in skill_counter.most_common(top_k):
        priority = "high" if count >= 3 else "medium" if count == 2 else "low"
        learning_suggestions.append({
            "skill": skill,
            "reason": f"Xuất hiện trong {count} job phù hợp trên ChromaDB và đang là khoảng trống so với CV.",
            "related_jobs_count": count,
            "example_jobs": skill_examples[skill][:3],
            "priority": priority,
        })

    market_summary = {
        "top_jobs_found": len(matched_jobs),
        "cv_skills": sorted(cv_skills),
        "most_requested_skills": [
            {"skill": skill, "count": count}
            for skill, count in skill_counter.most_common(10)
        ],
    }

    return {
        "matched_jobs": matched_jobs,
        "learning_suggestions": learning_suggestions,
        "market_summary": market_summary,
    }


def generate_question_suggestions(context: str, cv_info: Optional[Dict] = None, job_info: Optional[Dict] = None) -> List[Dict[str, str]]:
    """
    Tạo gợi ý câu hỏi dựa trên context
    
    Args:
        context: Context hiện tại (cv_uploaded, viewing_job, chatting)
        cv_info: Thông tin CV (optional)
        job_info: Thông tin job (optional)
        
    Returns:
        List các câu hỏi gợi ý
    """
    suggestions = []
    
    if context == "cv_uploaded":
        suggestions = [
            {
                "question": "CV của tôi có điểm mạnh gì?",
                "category": "cv_analysis",
                "icon": ""
            },
            {
                "question": "Tôi nên cải thiện kỹ năng gì để tăng cơ hội?",
                "category": "improvement",
                "icon": ""
            },
            {
                "question": "Mức lương tôi có thể mong đợi là bao nhiêu?",
                "category": "salary",
                "icon": ""
            },
            {
                "question": "Có job nào phù hợp với tôi không?",
                "category": "job_match",
                "icon": ""
            },
            {
                "question": "CV của tôi thiếu gì so với thị trường?",
                "category": "gap_analysis",
                "icon": ""
            }
        ]
    
    elif context == "viewing_job":
        job_title = job_info.get('job_title', 'công việc này') if job_info else 'công việc này'
        suggestions = [
            {
                "question": f"Tôi có phù hợp với vị trí {job_title} không?",
                "category": "job_fit",
                "icon": ""
            },
            {
                "question": "Tôi cần chuẩn bị gì để ứng tuyển?",
                "category": "preparation",
                "icon": ""
            },
            {
                "question": "Mức lương của vị trí này có hợp lý không?",
                "category": "salary",
                "icon": ""
            },
            {
                "question": "Công ty này có uy tín không?",
                "category": "company",
                "icon": ""
            }
        ]
    
    elif context == "chatting":
        suggestions = [
            {
                "question": "Tôi nên học skill gì tiếp theo?",
                "category": "learning",
                "icon": ""
            },
            {
                "question": "Làm thế nào để tăng cơ hội được tuyển?",
                "category": "tips",
                "icon": ""
            },
            {
                "question": "Có khóa học nào phù hợp với tôi?",
                "category": "courses",
                "icon": ""
            }
        ]
    
    else:
        # Default suggestions
        suggestions = [
            {
                "question": "Tôi có thể hỏi gì?",
                "category": "general",
                "icon": ""
            }
        ]
    
    return suggestions