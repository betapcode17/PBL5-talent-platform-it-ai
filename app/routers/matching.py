# app/routers/matching.py
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query, Request
from typing import List, Optional
from models.core import MatchedJob
from models.responses import (
    ApplicationItem, MatchInput, MatchResponse, ApplyJobInput, ApplicationResponse, ApplicationsResponse
)
from services.chroma_utils import index_cv_extracts
from services.rag_matching import match_cv
from services.db_utils import (
    get_all_cvs, get_db_connection, get_cached_matches, get_filtered_jobs, insert_cv_record, insert_match_log,
    insert_application, get_applications_by_cv, check_application_exists
)
from services.ai_analysis import generate_why_match
from utils.date_utils import normalize_date
from utils.validators import _to_int_job_id
from utils.pdf_parser import parse_cv_input_string
import time
import uuid
import logging
import json
import re

router = APIRouter()

@router.post("/", response_model=MatchResponse)
async def match_cv_endpoint(input: MatchInput, request: Request):
    """Khớp CV với công việc, sử dụng lọc trước và post-processing."""
    start_time = time.time()
    valid_keys = {"job_type", "work_location", "experience", "education", "skills", "deadline_after"}
    cleaned_filters = {k: v for k, v in input.filters.items() if k in valid_keys}
    if cleaned_filters != input.filters:
        logging.warning(f"Bộ lọc không hợp lệ được bỏ qua: {input.filters}. Sử dụng: {cleaned_filters}")
    session_id = input.session_id or str(uuid.uuid4())
    model_name = input.model.value
    cv_id = None
    try:
        # Lấy CV
        cv_start = time.time()
        if input.cv_id:
            cv_id = input.cv_id
            if not isinstance(cv_id, int):
                raise HTTPException(status_code=400, detail="cv_id must be an integer")
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, filename, cv_info_json FROM cv_store WHERE id = ?", (input.cv_id,))
                cv = cursor.fetchone()
                if not cv:
                    raise HTTPException(status_code=404, detail=f"CV với ID {input.cv_id} không tìm thấy")
                cv_info = json.loads(cv["cv_info_json"])
        elif input.cv_input:
            cv_id = input.cv_id or str(uuid.uuid4())
            cv_info = parse_cv_input_string(input.cv_input)
            cv_id = insert_cv_record("manual_input", cv_info)
            await index_cv_extracts(cv_info["skills"], cv_info["career_objective"], cv_info["education"], cv_id)
        else:
            cvs = get_all_cvs()
            if not cvs:
                raise HTTPException(status_code=404, detail="Không tìm thấy CV nào")
            cv = cvs[0]
            cv_id = cv["id"]
            cv_info = json.loads(cv["cv_info_json"])
        logging.info(f"✅ Lấy CV {cv_id} thành công ({time.time() - cv_start:.2f}s)")
        # Chuẩn hóa education và experience
        input_start = time.time()
        skills = cv_info.get("skills", [])
        aspirations = cv_info.get("career_objective", "")
        education = cv_info.get("education", [])
        experience = cv_info.get("experience", [])
        for edu in education:
            try:
                edu["start_date"] = normalize_date(edu.get("start_date", ""))
                edu["end_date"] = normalize_date(edu.get("end_date", ""))
                edu["school"] = edu.get("school") or "Unknown"
                edu["degree"] = edu.get("degree") or "Unknown"
                edu["major"] = edu.get("major") or "Unknown"
            except Exception as e:
                logging.warning(f"Error normalizing education: {str(e)}")
                edu["start_date"] = ""
                edu["end_date"] = ""
                edu["school"] = "Unknown"
                edu["degree"] = "Unknown"
                edu["major"] = "Unknown"
        for exp in experience:
            try:
                exp["start_date"] = normalize_date(exp.get("start_date", ""))
                exp["end_date"] = normalize_date(exp.get("end_date", ""))
                exp["company"] = exp.get("company") or "Unknown"
                exp["title"] = exp.get("title") or "Unknown"
                exp["description"] = exp.get("description") or "No description provided"
            except Exception as e:
                logging.warning(f"Error normalizing experience: {str(e)}")
                exp["start_date"] = ""
                exp["end_date"] = ""
                exp["company"] = "Unknown"
                exp["title"] = "Unknown"
                exp["description"] = "No description provided"
        experience_summary = "\n".join([
            f"Project: {exp['title']} - {exp['description'][:200] + '...' if len(exp['description']) > 200 else exp['description']}"
            for exp in experience
        ]) if experience else "No experience provided"
        education_summary = "\n".join([
            f"Degree: {edu['degree']} at {edu['school']} ({edu['start_date']}-{edu['end_date']})"
            for edu in education
        ]) if education else "No education provided"
        aspirations_summary = aspirations if aspirations else "No career objective provided"
        cv_input = {
            "skills": skills,
            "aspirations": aspirations_summary,
            "experience": experience_summary,
            "education": education_summary,
            "cv_id": cv_id
        }
        # Lấy filtered_job_ids
        filtered_job_ids = get_filtered_jobs(cleaned_filters)
        suggestions = []
        if filtered_job_ids is None:
            suggestions = [{"skill_or_experience": "N/A", "suggestion": "No filters applied or no jobs matched, showing best matches from all jobs."}]
        else:
            logging.info(f"✅ Lọc được {len(filtered_job_ids)} jobs")
        # Kiểm tra cache trước
        cached_jobs = get_cached_matches(cv_id)
        if cached_jobs and not cleaned_filters:
            # Có cache và không có filters → Dùng cache
            logging.info(f"🚀 Sử dụng cached matches cho CV {cv_id} (skip RAG)")
            matched_jobs_all = cached_jobs
            suggestions = []
        else:
            # Chạy RAG với match_cv
            try:
                invoke_start = time.time()
                result = await match_cv(cv_input, filtered_job_ids, session_id)
                logging.info(f"✅ Match CV hoàn tất ({time.time() - invoke_start:.2f}s)")
                if not result or not isinstance(result, dict):
                    logging.error("match_cv trả về kết quả rỗng")
                    return MatchResponse(
                        name=cv_info.get("name", ""),
                        email=cv_info.get("email", ""),
                        phone=cv_info.get("phone", ""),
                        cv_skills=skills,
                        career_objective=aspirations,
                        education=education,
                        experience=experience,
                        matched_jobs=[],
                        suggestions=[{"skill_or_experience": "N/A", "suggestion": "Failed to match jobs"}],
                        session_id=session_id,
                        model=input.model
                    )
                answer = result
                # Lấy TẤT CẢ matched_jobs (có thể lên đến 20)
                matched_jobs_all = answer.get("matched_jobs", [])
                suggestions = answer.get("suggestions", []) or suggestions
                logging.info(f"📊 Nhận được {len(matched_jobs_all)} jobs từ RAG")
            except Exception as e:
                logging.error(f"❌ Lỗi match_cv: {e}")
                raise
        # Validate matched_jobs_all
        if not isinstance(matched_jobs_all, list):
            logging.error(f"matched_jobs không phải danh sách: {matched_jobs_all}")
            matched_jobs_all = []
        for job in matched_jobs_all[:]:
            if not isinstance(job, dict) or "job_id" not in job or "match_score" not in job:
                logging.warning(f"Job thiếu job_id hoặc match_score: {job}. Skipping.")
                matched_jobs_all.remove(job)
                continue
            for field in ["matched_skills", "matched_aspirations", "matched_experience", "matched_education", "skills"]:
                if not isinstance(job.get(field, []), list):
                    logging.error(f"Trường {field} không phải danh sách trong job: {job}")
                    job[field] = []
        if not isinstance(suggestions, list):
            logging.error(f"suggestions không phải danh sách: {suggestions}")
            suggestions = []
        # Post-processing: chuẩn hóa job_id -> int, enrich từ DB
        # 1) Chuẩn hóa danh sách job_id cho TẤT CẢ jobs
        job_ids: List[int] = []
        for job in matched_jobs_all:
            jid_int = _to_int_job_id(job.get("job_id"))
            if jid_int is None:
                logging.warning(f"Invalid job_id skipped: {job.get('job_id')}")
                continue
            job_ids.append(jid_int)
        # 2) Lấy chi tiết từ DB
        if not job_ids:
            job_details = []
        else:
            from services.db_utils import get_job_details
            job_details = get_job_details(job_ids)
        # 3) Map chi tiết theo INT key (quan trọng)
        job_details_dict = {int(job.job_id): job for job in job_details if getattr(job, "job_id", None) is not None}
        # 4) Enrich TẤT CẢ jobs (lên đến 20 jobs)
        enriched_all_jobs = []
        for job in matched_jobs_all:
            jid_int = _to_int_job_id(job.get("job_id"))
            if jid_int is None:
                continue
            detail = job_details_dict.get(jid_int)
            if not detail:
                logging.warning(f"Job ID {job.get('job_id')} not found in job_details")
                continue
            # match_score có thể là 0..1 hoặc phần trăm >1 -> đưa về 0..1
            ms = job.get("match_score", 0.0)
            try:
                ms = float(ms)
                if ms > 1.0:  # ví dụ 62 -> 0.62
                    ms = ms / 100.0
                ms = max(0.0, min(1.0, ms))
            except Exception:
                ms = 0.0
            enriched_all_jobs.append(
                MatchedJob(
                    job_id=str(jid_int),
                    job_title=job.get("job_title") or getattr(detail, "job_title", ""),
                    job_url=(job.get("job_url") or getattr(detail, "job_url", "")),
                    match_score=ms,
                    matched_skills=job.get("matched_skills") or [],
                    matched_aspirations=job.get("matched_aspirations") or [],
                    matched_experience=job.get("matched_experience") or [],
                    matched_education=job.get("matched_education") or [],
                    work_location=getattr(detail, "work_location", None),
                    salary=getattr(detail, "salary", None),
                    deadline=getattr(detail, "deadline", None),
                    benefits=getattr(detail, "benefits", None),
                    job_type=getattr(detail, "work_type", None),
                    experience_required=getattr(detail, "experience", None),
                    education_required=getattr(detail, "education", None),
                    company_name=getattr(detail, "name", None),
                    skills=getattr(detail, "skills", None),
                    why_match=job.get("why_match", None),  # AI-generated reason
                    job_description=getattr(detail, "job_description", None),
                )
            )
        # 5) Lưu TẤT CẢ jobs vào cache (20 jobs)
        safe_all_jobs = [
            {
                "job_id": job.job_id,
                "job_title": job.job_title,
                "job_url": job.job_url,
                "match_score": job.match_score,
                "matched_skills": job.matched_skills,
                "matched_aspirations": job.matched_aspirations,
                "matched_experience": job.matched_experience,
                "matched_education": job.matched_education,
                "work_location": job.work_location,
                "salary": job.salary,
                "deadline": job.deadline,
                "benefits": job.benefits,
                "job_type": job.job_type,
                "experience_required": job.experience_required,
                "education_required": job.education_required,
                "company_name": job.company_name,
                "skills": job.skills,
                "why_match": job.why_match,
                "job_description": job.job_description
            }
            for job in enriched_all_jobs
        ]
        # Lưu cache (20 jobs)
        if not cached_jobs:  # Chỉ lưu nếu không dùng cache
            insert_match_log(session_id, cv_id, safe_all_jobs)
            logging.info(f"💾 Đã cache {len(safe_all_jobs)} jobs cho CV {cv_id}")
        # 6) Trả về TOP 5 jobs
        top_5_jobs = enriched_all_jobs[:5]
        logging.info(f"✅ Hoàn tất! CV {cv_id} matched {len(enriched_all_jobs)} jobs, trả về top {len(top_5_jobs)} | Tổng thời gian: {time.time() - start_time:.2f}s")
        return MatchResponse(
            name=cv_info.get("name", ""),
            email=cv_info.get("email", ""),
            phone=cv_info.get("phone", ""),
            cv_skills=skills,
            career_objective=aspirations,
            education=education,
            experience=experience,
            matched_jobs=top_5_jobs,  # Chỉ trả về top 5
            suggestions=suggestions,
            session_id=session_id,
            model=input.model
        )
    except Exception as e:
        logging.error(f"Lỗi khi khớp CV {cv_id}: {str(e)}")
        return MatchResponse(
                name=cv_info.get("name", ""),
                email=cv_info.get("email", ""),
                phone=cv_info.get("phone", ""),
                cv_skills=skills,
                career_objective=aspirations,
                education=education,
                experience=experience,
                matched_jobs=[],
                suggestions=[{"skill_or_experience": "N/A", "suggestion": f"Failed to match jobs: {str(e)}"}],
                session_id=session_id,
                model=input.model
            )
    except json.JSONDecodeError as e:
        logging.error(f"JSON không hợp lệ trong CV {cv_id if cv_id else 'chưa xác định'}: {str(e)}")
        raise HTTPException(status_code=500, detail="Dữ liệu JSON CV không hợp lệ")
    except Exception as e:
        logging.error(f"Lỗi khi truy cập CV {cv_id if cv_id else 'chưa xác định'}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Không thể truy cập dữ liệu CV: {str(e)}")

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