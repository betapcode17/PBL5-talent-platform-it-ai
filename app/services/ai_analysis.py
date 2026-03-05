"""
AI Analysis Functions - Phân tích CV và Jobs bằng Gemini AI
"""
import logging
import json
from typing import Dict, List, Optional, Any
from langchain_google_genai import ChatGoogleGenerativeAI

from services.api_key_manager import get_next_api_key
from prompts import cv_analysis_prompt, cv_improvement_prompt  # Import prompts

# Initialize Gemini model with API key rotation
def get_llm():
    """
    Get LLM instance with rotated API key
    Uses gemini-2.5-flash (stable, higher quota than 2.0-flash-exp)
    """
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=get_next_api_key(),
        temperature=0.3
    )

# Legacy global instance (for backward compatibility)
llm = get_llm()

async def analyze_cv_insights(cv_info: Dict) -> Dict[str, Any]:
    """
    Phân tích CV chuyên sâu - Đánh giá chất lượng, điểm mạnh/yếu
    
    Args:
        cv_info: Thông tin CV đã parse
        
    Returns:
        Dict chứa quality_score, strengths, weaknesses, completeness, market_fit
    """
    try:
        # Use fresh LLM instance with rotated API key
        llm_instance = get_llm()
        
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
        
        # Invoke prompt
        response = await llm_instance.ainvoke(cv_analysis_prompt.format(**formatted_input))
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

        logging.info(f"✅ Phân tích CV thành công: quality_score={result.get('quality_score')}, completeness={result.get('completeness_score')}")
        return result
        
    except json.JSONDecodeError as e:
        logging.error(f"❌ Lỗi parse JSON từ Gemini: {e}")
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
        logging.error(f"❌ Lỗi phân tích CV: {e}")
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
        # Use fresh LLM instance with rotated API key
        llm_instance = get_llm()
        
        # Format input for prompt
        formatted_input = {
            "skills": ', '.join(cv_info.get('skills', [])) if cv_info.get('skills') else 'Chưa có',
            "experience_count": len(cv_info.get('experience', [])),
            "education_count": len(cv_info.get('education', [])),
            "career_objective": cv_info.get('career_objective', 'Chưa có')[:200] + '...' if cv_info.get('career_objective') else 'Chưa có',
            "weaknesses": ', '.join(insights.get('weaknesses', [])),
            "missing_sections": ', '.join(insights.get('missing_sections', [])),
            "has_portfolio": insights.get('has_portfolio', False),
            "has_certifications": insights.get('has_certifications', False),
            "has_projects": insights.get('has_projects', False)
        }
        
        # Invoke prompt
        response = await llm_instance.ainvoke(cv_improvement_prompt.format(**formatted_input))
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

        logging.info(f"✅ Tạo {len(improvements)} gợi ý cải thiện")
        return improvements
        
    except json.JSONDecodeError as e:
        logging.error(f"❌ Lỗi parse JSON: {e}")
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
        logging.error(f"❌ Lỗi tạo gợi ý: {e}")
        raise


async def generate_why_match(cv_skills: List[str], job_skills: List[str], job_title: str) -> str:
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
                "icon": "💪"
            },
            {
                "question": "Tôi nên cải thiện kỹ năng gì để tăng cơ hội?",
                "category": "improvement",
                "icon": "📈"
            },
            {
                "question": "Mức lương tôi có thể mong đợi là bao nhiêu?",
                "category": "salary",
                "icon": "💰"
            },
            {
                "question": "Có job nào phù hợp với tôi không?",
                "category": "job_match",
                "icon": "🎯"
            },
            {
                "question": "CV của tôi thiếu gì so với thị trường?",
                "category": "gap_analysis",
                "icon": "🔍"
            }
        ]
    
    elif context == "viewing_job":
        job_title = job_info.get('job_title', 'công việc này') if job_info else 'công việc này'
        suggestions = [
            {
                "question": f"Tôi có phù hợp với vị trí {job_title} không?",
                "category": "job_fit",
                "icon": "🎯"
            },
            {
                "question": "Tôi cần chuẩn bị gì để ứng tuyển?",
                "category": "preparation",
                "icon": "📝"
            },
            {
                "question": "Mức lương của vị trí này có hợp lý không?",
                "category": "salary",
                "icon": "💰"
            },
            {
                "question": "Công ty này có uy tín không?",
                "category": "company",
                "icon": "🏢"
            }
        ]
    
    elif context == "chatting":
        suggestions = [
            {
                "question": "Tôi nên học skill gì tiếp theo?",
                "category": "learning",
                "icon": "📚"
            },
            {
                "question": "Làm thế nào để tăng cơ hội được tuyển?",
                "category": "tips",
                "icon": "💡"
            },
            {
                "question": "Có khóa học nào phù hợp với tôi?",
                "category": "courses",
                "icon": "🎓"
            }
        ]
    
    else:
        # Default suggestions
        suggestions = [
            {
                "question": "Tôi có thể hỏi gì?",
                "category": "general",
                "icon": "❓"
            }
        ]
    
    return suggestions