# app/routers/matching.py
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query, Request
from typing import List, Optional
from services.match_explain import build_why_match
from models.core import MatchedJob
from models.responses import (
    ApplicationItem, MatchExplanation, MatchInput, MatchResponse, ApplyJobInput, ApplicationResponse, ApplicationsResponse
)
from services.chroma_utils import index_cv_extracts
from services.rag_matching import match_cv
from services.db_utils import (
    get_all_cvs, get_db_connection, get_cached_matches, get_filtered_jobs, get_jobs_details_by_ids, insert_cv_record, insert_match_log,
    insert_application, get_applications_by_cv, check_application_exists
)
from services.ai_analysis import generate_why_match
from utils.date_utils import normalize_date, normalize_deadline
from utils.validators import _to_int_job_id
from utils.pdf_parser import parse_cv_input_string
import time
import uuid
import logging
import json
import re

router = APIRouter()

from services.match_explain import normalize_explanation, build_why_match

@router.post("/", response_model=MatchResponse)
async def match_cv_endpoint(input: MatchInput, request: Request):
    start_time = time.time()
    session_id = input.session_id or str(uuid.uuid4())
    cv_id = None

    try:
        # --------------------------------------------------
        # 1. CLEAN FILTERS
        # --------------------------------------------------
        valid_keys = {
            "job_type", "work_location", "experience",
            "education", "skills", "deadline_after"
        }
        cleaned_filters = {
            k: v for k, v in (input.filters or {}).items()
            if k in valid_keys
        }

        # --------------------------------------------------
        # 2. LOAD CV
        # --------------------------------------------------
        if input.cv_id:
            with get_db_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT id, cv_info_json FROM cv_store WHERE id = ?",
                    (input.cv_id,)
                )
                row = cur.fetchone()
                if not row:
                    raise HTTPException(404, "CV not found")
                cv_id = row["id"]
                cv_info = json.loads(row["cv_info_json"])
        else:
            cvs = get_all_cvs()
            if not cvs:
                raise HTTPException(404, "No CV found")
            cv = cvs[0]
            cv_id = cv["id"]
            cv_info = json.loads(cv["cv_info_json"])

        # --------------------------------------------------
        # 3. PREPARE CV INPUT
        # --------------------------------------------------
        skills = cv_info.get("skills", [])
        aspirations = cv_info.get("career_objective", "")
        experience = cv_info.get("experience", [])
        education = cv_info.get("education", [])

        cv_input = {
            "skills": skills,
            "aspirations": aspirations,
            "experience": "\n".join(
                f"{e.get('title', '')}: {e.get('description', '')[:200]}"
                for e in experience
            ),
            "education": "\n".join(
                f"{e.get('degree', '')} at {e.get('school', '')}"
                for e in education
            ),
            "cv_id": cv_id
        }

        # --------------------------------------------------
        # 4. MATCH
        # --------------------------------------------------
        filtered_job_ids = get_filtered_jobs(cleaned_filters)
        # Ensure filtered_job_ids is always a list of integers
        if not isinstance(filtered_job_ids, list):
            logging.warning(f"⚠️ filtered_job_ids không phải list: {type(filtered_job_ids)} = {filtered_job_ids}")
            filtered_job_ids = []
        # Validate all items are integers
        filtered_job_ids = [int(jid) for jid in filtered_job_ids if isinstance(jid, (int, str)) and str(jid).isdigit()]
        logging.info(f"✅ Filtered job IDs: {len(filtered_job_ids)} jobs")
        
        # 🔍 DEBUG: Log CV input
        logging.info(f"🔍 CV Skills: {len(cv_input.get('skills', []))} skills")
        logging.info(f"🔍 CV Experience length: {len(cv_input.get('experience', ''))} chars")
        
        result = await match_cv(cv_input, filtered_job_ids, session_id)
        
        # 🔍 DEBUG: Log result
        logging.info(f"🔍 RAG returned {len(result.get('matched_jobs', []))} matched jobs")
        if not result.get("matched_jobs"):
            logging.error(f"❌ Không có matched_jobs từ RAG. Result keys: {result.keys()}")

        safe_jobs = [
            j for j in result.get("matched_jobs", [])
            if isinstance(j, dict) and "job_id" in j
        ]
        
        logging.info(f"🔍 After filtering: {len(safe_jobs)} safe jobs")

        # Filter out None and invalid job IDs
        job_ids = [_to_int_job_id(j["job_id"]) for j in safe_jobs]
        job_ids = [jid for jid in job_ids if jid is not None and isinstance(jid, int)]
        
        if not job_ids:
            logging.warning("⚠️ Không có job_id hợp lệ sau khi lọc")
            job_details = []
        else:
            logging.info(f"✅ Lấy chi tiết cho {len(job_ids)} jobs")
            job_details = get_jobs_details_by_ids(job_ids)
        job_map = {int(j["id"]): j for j in job_details}

        enriched_all_jobs = []

        for job in safe_jobs:
            jid = _to_int_job_id(job.get("job_id"))
            detail = job_map.get(jid)
            if not jid or not detail:
                continue

            # score
            ms = float(job.get("match_score", 0))
            if ms > 1:
                ms /= 100
            ms = max(0.0, min(1.0, ms))

            # 🔥 FIX EXPLANATION
            raw_expl = job.get("explanation")
            normalized_expl = normalize_explanation(raw_expl)

            explanation_model = MatchExplanation(**normalized_expl)
            why_match = build_why_match(normalized_expl)

            enriched_all_jobs.append(
                MatchedJob(
                    job_id=str(jid),
                    job_title=detail.get("job_title", ""),
                    job_url=detail.get("job_url", ""),
                    work_location=detail.get("work_location", ""),
                    salary=detail.get("salary", ""),
                    deadline=normalize_deadline(detail.get("deadline")),
                    benefits=detail.get("benefits", ""),
                    job_type=detail.get("work_type", ""),
                    experience_required=detail.get("experience", ""),
                    education_required=detail.get("education", ""),
                    company_name=detail.get("name", ""),
                    skills=detail.get("skills", "").split(","),
                    match_score=ms,
                    explanation=explanation_model,
                    why_match=why_match,
                    job_description=detail.get("job_description", "")
                )
            )

        enriched_all_jobs.sort(key=lambda x: x.match_score, reverse=True)

        return MatchResponse(
            name=cv_info.get("name", ""),
            email=cv_info.get("email", ""),
            phone=cv_info.get("phone", ""),
            cv_skills=skills,
            career_objective=aspirations,
            education=education,
            experience=experience,
            matched_jobs=enriched_all_jobs,
            suggestions=result.get("suggestions", []),
            session_id=session_id,
            model=input.model
        )

    except Exception as e:
        logging.exception("❌ match_cv_endpoint failed")
        return MatchResponse(
            name="",
            email="",
            phone="",
            cv_skills=[],
            career_objective="",
            education=[],
            experience=[],
            matched_jobs=[],
            suggestions=[{
                "skill_or_experience": "N/A",
                "suggestion": f"Failed to match jobs: {str(e)}"
            }],
            session_id=session_id,
            model=input.model
        )


@router.post("/apply", response_model=ApplicationResponse)
async def apply_job_endpoint(input: ApplyJobInput):
    """
    Ứng tuyển job - Lưu lại hành động ứng tuyển
    Endpoint mới để tracking việc ứng tuyển của user.
    """
    try:
        cv_id = input.cv_id
        job_id = input.job_id
        cover_letter = input.cover_letter
        status = input.status
        # Kiểm tra CV tồn tại
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM cv_store WHERE id = ?", (cv_id,))
            if not cursor.fetchone():
                raise HTTPException(status_code=404, detail=f"CV {cv_id} không tìm thấy")
        # Kiểm tra job tồn tại
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM job_store WHERE id = ?", (job_id,))
            if not cursor.fetchone():
                raise HTTPException(status_code=404, detail=f"Job {job_id} không tìm thấy")
        # Kiểm tra đã apply chưa
        if check_application_exists(cv_id, job_id):
            raise HTTPException(status_code=400, detail=f"Đã ứng tuyển job {job_id} rồi")
        # Lưu application
        app_id = insert_application(cv_id, job_id, cover_letter, status)
        logging.info(f"✅ CV {cv_id} đã ứng tuyển job {job_id}")
        return ApplicationResponse(
            application_id=app_id,
            cv_id=cv_id,
            job_id=job_id,
            status=status,
            applied_at=datetime.now().isoformat(),
            message=f"Đã ứng tuyển job {job_id} thành công"
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"❌ Lỗi ứng tuyển: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi ứng tuyển: {str(e)}")

@router.get("/applications/{cv_id}", response_model=ApplicationsResponse)
async def get_applications_endpoint(
    cv_id: int,
    status: Optional[str] = Query(None, description="Lọc theo status (applied, pending, accepted, rejected)")
):
    """
    Lấy lịch sử ứng tuyển của CV
    Endpoint mới để xem các job đã apply.
    """
    try:
        # Kiểm tra CV tồn tại
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM cv_store WHERE id = ?", (cv_id,))
            if not cursor.fetchone():
                raise HTTPException(status_code=404, detail=f"CV {cv_id} không tìm thấy")
        # Lấy applications
        applications = get_applications_by_cv(cv_id, status)
        app_items = [
            ApplicationItem(
                id=app['id'],
                cv_id=app['cv_id'],
                job_id=app['job_id'],
                job_title=app['job_title'],
                company_url=app['company_url'],
                salary=app['salary'],
                work_location=app['work_location'],
                status=app['status'],
                applied_at=app['applied_at']
            )
            for app in applications
        ]
        logging.info(f"✅ Lấy {len(app_items)} applications cho CV {cv_id}")
        return ApplicationsResponse(
            cv_id=cv_id,
            total=len(app_items),
            applications=app_items
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"❌ Lỗi lấy applications: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi lấy applications: {str(e)}")