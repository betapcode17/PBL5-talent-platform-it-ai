# app/routers/cv.py
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import Dict, List
from app.models.core import DocumentInfo, DeleteFileRequest
from app.models.responses import (
    CVInsightsResponse,
    CVImproveResponse,
    ImprovementSuggestion,
    CVFileAnalysisResponse,
    CVJobMatchInsight,
    LearningRecommendation,
)


from app.services.ai_analysis import analyze_cv_insights, generate_cv_improvements, analyze_cv_against_jobs
from app.utils.pdf_parser import extract_text_from_pdf, extract_cv_info
from app.utils.date_utils import normalize_date
import os
import json
import logging
from datetime import datetime

router = APIRouter()


# Database helpers (may be monkeypatched in tests or implemented elsewhere).
def insert_cv_record(filename, cv_info, file_data):
    """Store the CV file and metadata and return a cv_id.

    This is a placeholder implementation; the real DB integration should
    provide this function or tests may monkeypatch it.
    """
    raise NotImplementedError("insert_cv_record is not implemented")


def save_cv_insights(cv_id, insights):
    """Persist CV insights for later retrieval.

    Placeholder — to be provided by DB layer or test monkeypatch.
    """
    raise NotImplementedError("save_cv_insights is not implemented")


def delete_cv_record(cv_id):
    """Delete a previously inserted CV record. Placeholder implementation."""
    raise NotImplementedError("delete_cv_record is not implemented")


def _build_cv_insights_response(cv_id: int, insights: Dict, analyzed_at: str) -> CVInsightsResponse:
    return CVInsightsResponse(
        cv_id=cv_id,
        quality_score=insights.get('quality_score', 5.0),
        completeness={
            "has_portfolio": insights.get('has_portfolio', False),
            "has_certifications": insights.get('has_certifications', False),
            "has_projects": insights.get('has_projects', False),
            "missing_sections": insights.get('missing_sections', []),
        },
        market_fit={
            "skill_match_rate": insights.get('market_fit_score', 0.5),
            "experience_level": insights.get('experience_level', 'Unknown'),
            "salary_range": insights.get('salary_range', 'N/A'),
            "competitive_score": insights.get('competitive_score', 5.0),
        },
        strengths=insights.get('strengths', []),
        weaknesses=insights.get('weaknesses', []),
        last_analyzed=analyzed_at,
    )


def _fallback_cv_insights(cv_info: Dict) -> Dict:
    skills = cv_info.get("skills", []) or []
    experience = cv_info.get("experience", []) or []
    education = cv_info.get("education", []) or []
    # Point 2: has_projects should NOT be inferred from experience; check cv_info.has_projects
    has_projects_actual = cv_info.get("has_projects", False)

    strengths = []
    if skills:
        # Point 3: Don't be generic; be specific (not "extracted N skills")
        tech_skills = [s for s in skills if len(s) > 2]
        if tech_skills:
            strengths.append(f"Technical stack includes: {', '.join(tech_skills[:5])}")
    if experience:
        strengths.append(f"Demonstrated {len(experience)} positions with hands-on experience")
    if education:
        strengths.append(f"Formal education: {len(education)} degree(s)/qualification(s)")

    weaknesses = []
    if not skills:
        weaknesses.append("Missing technical skills section or detail")
    if not experience:
        weaknesses.append("No professional work experience documented")
    if not education:
        weaknesses.append("Education/qualification information not found")
    if not cv_info.get("career_objective"):
        weaknesses.append("No career objective, summary, or seeking statement")
    # Point 1: Don't report missing projects if has_projects_actual is True
    if not has_projects_actual and "projects" not in weaknesses:
        weaknesses.append("No projects or case studies documented")
    if not weaknesses:
        weaknesses.append("Recommend adding quantified metrics and specific project outcomes")

    return {
        "quality_score": 5.0 if not skills else min(8.5, 4.5 + 0.4 * len(skills)),  # Point 8: cap at 8.5 for intern
        "completeness_score": 0.5,
        "has_portfolio": False,
        "has_certifications": False,
        "has_projects": has_projects_actual,  # Point 2: use actual value, not inferred
        "missing_sections": [
            section for section, present in (
                ("portfolio", False),
                ("certifications", False),
                ("projects", not has_projects_actual),  # Point 1: use actual has_projects
            ) if not present
        ],
        "market_fit_score": 0.5,
        "experience_level": "Junior" if (skills and has_projects_actual) else "Intern",  # Point 7: infer from signals
        "salary_range": "N/A",
        "competitive_score": 5.0,
        "strengths": strengths or ["CV structure recognized"],
        "weaknesses": weaknesses,
    }


def _fallback_cv_insights_from_text(cv_text: str) -> Dict:
    words = len(cv_text.split()) if cv_text else 0
    has_email = "@" in cv_text
    has_phone = any(ch.isdigit() for ch in cv_text)
    has_long_text = words >= 80

    strengths = []
    if has_email:
        strengths.append("CV có thông tin liên hệ rõ ràng")
    if has_phone:
        strengths.append("CV có số điện thoại hoặc chuỗi số nhận diện")
    if has_long_text:
        strengths.append("CV có đủ nội dung để trích xuất và phân tích sơ bộ")

    weaknesses = [
        "Hệ thống không trích xuất được phần kỹ năng hoặc mục tiêu nghề nghiệp rõ ràng",
        "Nên trình bày kỹ năng, kinh nghiệm và mục tiêu nghề nghiệp theo từng mục riêng",
        "Có thể cần tối ưu lại định dạng PDF nếu CV là bản scan hoặc ảnh",
    ]

    return {
        "quality_score": 4.0 if has_long_text else 2.5,
        "completeness_score": 0.25 if has_long_text else 0.15,
        "has_portfolio": False,
        "has_certifications": False,
        "has_projects": False,
        "missing_sections": ["skills", "career_objective", "experience", "education"],
        "market_fit_score": 0.2 if has_long_text else 0.1,
        "experience_level": "Unknown",
        "salary_range": "N/A",
        "competitive_score": 3.0 if has_long_text else 2.0,
        "strengths": strengths or ["CV đã được nhận diện ở mức cơ bản"],
        "weaknesses": weaknesses,
    }


def _has_usable_cv_content(cv_info: Dict) -> bool:
    skills = cv_info.get("skills", []) or []
    objective = str(cv_info.get("career_objective") or "").strip()
    experience = cv_info.get("experience", []) or []
    education = cv_info.get("education", []) or []
    return bool(skills or objective or experience or education)




@router.post("/analyze", response_model=CVFileAnalysisResponse)
async def analyze_uploaded_cv(file: UploadFile = File(...), top_k: int = 5):
    """Upload CV PDF, phân tích ngay và gợi ý nên học gì dựa trên job data trong ChromaDB."""
    if not file.filename.lower().endswith('.pdf'):  # type: ignore
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    temp_file_path = f"temp_{file.filename}"
    try:
        file_data = await file.read()
        if len(file_data) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size exceeds 10MB")

        with open(temp_file_path, "wb") as buffer:
            buffer.write(file_data)

        cv_text = extract_text_from_pdf(temp_file_path)
        cv_info = extract_cv_info(cv_text)
        skills = cv_info.get("skills", [])
        aspirations = cv_info.get("career_objective", "")
        education = cv_info.get("education", [])
        experience = cv_info.get("experience", [])
        usable_content = _has_usable_cv_content(cv_info)

        experience_summary = "\n".join([
            f"{exp.get('title', 'Unknown')} at {exp.get('company', 'Unknown')} ({exp.get('start_date', '')}-{exp.get('end_date', '')}): {exp.get('description', '')}"
            for exp in experience
        ]) if experience else "No experience provided"

        cv_id = insert_cv_record(file.filename, cv_info, file_data)  # type: ignore
        if not cv_id:
            raise HTTPException(status_code=500, detail="Failed to generate cv_id from database")

        # Chroma indexing intentionally disabled — skip indexing step.
        # Previously the system called `index_cv_extracts(...)` here to store CV
        # vectors in ChromaDB. That behavior is now disabled by design.

        try:
            if usable_content:
                insights_raw = await analyze_cv_insights(cv_info)
            else:
                raise ValueError("Insufficient structured CV content")
        except Exception as analysis_error:
            logging.warning(f"Falling back to default CV insights for {cv_id}: {analysis_error}")
            insights_raw = _fallback_cv_insights(cv_info) if usable_content else _fallback_cv_insights_from_text(cv_text)

        save_cv_insights(cv_id, insights_raw)
        insights_response = _build_cv_insights_response(cv_id, insights_raw, datetime.now().isoformat())

        try:
            job_analysis = await analyze_cv_against_jobs(cv_info, top_k=top_k) if usable_content else {"matched_jobs": [], "learning_suggestions": [], "market_summary": {"top_jobs_found": 0}}
        except Exception as job_error:
            logging.warning(f"Job analysis failed for CV {cv_id}: {job_error}")
            job_analysis = {"matched_jobs": [], "learning_suggestions": [], "market_summary": {}}

        # For this endpoint we intentionally do NOT return matched job listings.
        # The user wants only strengths, weaknesses and recommended learning items.
        learning_suggestions = [LearningRecommendation(**item) for item in job_analysis.get("learning_suggestions", [])]

        # Generate a concrete learning roadmap (LLM or heuristic fallback)
        try:
            from app.services.ai_analysis import generate_learning_roadmap
            roadmap = await generate_learning_roadmap(cv_info, insights_raw, job_analysis)
        except Exception as roadmap_error:
            logging.warning(f"Failed to generate learning roadmap for CV {cv_id}: {roadmap_error}")
            roadmap = []

        return CVFileAnalysisResponse(
            cv_id=cv_id,
            filename=file.filename, # type: ignore
            insights=insights_response,
            matched_jobs=[],  # omitted by design
            learning_suggestions=learning_suggestions,
            market_summary=job_analysis.get("market_summary", {}),
            extracted_text=cv_text[:500] + "..." if len(cv_text) > 500 else cv_text,  # Point 18: Truncate extracted_text in production
            learning_roadmap=roadmap, # type: ignore
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error analyzing CV {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze CV: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)



