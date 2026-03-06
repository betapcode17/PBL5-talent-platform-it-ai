# app/routers/jobs.py
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List, Set
from models.responses import JobSearchInput, JobSearchResponse, JobSearchResult
from services.db_utils import get_db_connection
from services.ai_analysis import get_llm
from services.chroma_utils import get_vectorstore
import json
import logging
from datetime import datetime, timedelta

router = APIRouter()

@router.get("/")
async def get_all_jobs_simple(limit: int = 100, offset: int = 0):
    """
    Lấy tất cả jobs (cho frontend dashboard và jobs listing)
    Khác với /jobs/search (cần cv_id và AI ranking),
    endpoint này chỉ đơn giản list jobs với pagination
    Parameters:
    - limit: Số lượng jobs tối đa (default: 100)
    - offset: Vị trí bắt đầu (default: 0)
    Returns:
    - jobs: List of job objects
    - total: Tổng số jobs trong database
    - limit: Limit được sử dụng
    - offset: Offset được sử dụng
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # Get total count
            cursor.execute("SELECT COUNT(*) as total FROM job_store")
            total = cursor.fetchone()["total"]
            # Get jobs with pagination
            cursor.execute(f"SELECT * FROM job_store LIMIT ? OFFSET ?", (limit, offset))
            jobs = [dict(row) for row in cursor.fetchall()]
            logging.info(f" Lấy {len(jobs)} jobs (total: {total}, limit: {limit}, offset: {offset})")
            return {
                "jobs": jobs,
                "total": total,
                "limit": limit,
                "offset": offset
            }
    except Exception as e:
        logging.error(f" Lỗi lấy jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi lấy jobs: {str(e)}")

@router.post("/search", response_model=JobSearchResponse)
async def search_jobs_endpoint(input: JobSearchInput):
    """
    Hybrid search: ket hop semantic search (ChromaDB) va keyword search (SQL LIKE).
    Ket qua duoc merge, deduplicate va rank theo combined score.
    """
    try:
        query = input.query
        filters = input.filters
        cv_id = input.cv_id
        limit = input.limit
        offset = input.offset
        logging.info(f"Hybrid search: query='{query}', filters={filters}, cv_id={cv_id}")

        # Lay CV info neu co cv_id (de AI ranking)
        cv_info = None
        if cv_id:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT cv_info_json FROM cv_store WHERE id = ?", (cv_id,))
                row = cursor.fetchone()
                if row:
                    cv_info = json.loads(row["cv_info_json"])

        # ---- 1. Keyword search (SQL LIKE) ----
        keyword_jobs: List[dict] = []
        keyword_ids: Set[int] = set()
        if query:
            kw_sql = "SELECT * FROM job_store WHERE 1=1"
            kw_params: list = []
            kw_sql += " AND (job_title LIKE ? OR job_description LIKE ? OR skills LIKE ?)"
            search_term = f"%{query}%"
            kw_params.extend([search_term, search_term, search_term])
            kw_sql, kw_params = _apply_filters(kw_sql, kw_params, filters)
            kw_sql += " LIMIT ?"
            kw_params.append(limit * 2)
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(kw_sql, kw_params)
                keyword_jobs = [dict(row) for row in cursor.fetchall()]
            keyword_ids = {j['id'] for j in keyword_jobs}

        # ---- 2. Semantic search (ChromaDB) ----
        semantic_jobs: List[dict] = []
        semantic_scores: Dict[int, float] = {}
        if query:
            try:
                vectorstore = get_vectorstore("jobs")
                sem_results = vectorstore.similarity_search_with_relevance_scores(
                    query, k=limit * 2
                )
                for doc, score in sem_results:
                    job_id = doc.metadata.get("job_id")
                    if job_id is not None:
                        try:
                            jid = int(job_id)
                            semantic_scores[jid] = float(score)
                        except (ValueError, TypeError):
                            pass
                # Fetch full job data for semantic-only results
                semantic_only_ids = [jid for jid in semantic_scores if jid not in keyword_ids]
                if semantic_only_ids:
                    placeholders = ','.join(['?' for _ in semantic_only_ids])
                    filter_sql = f"SELECT * FROM job_store WHERE id IN ({placeholders})"
                    filter_params: list = list(semantic_only_ids)
                    extra_conditions, extra_params = _build_filter_conditions(filters)
                    if extra_conditions:
                        filter_sql += " AND " + " AND ".join(extra_conditions)
                        filter_params.extend(extra_params)
                    with get_db_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute(filter_sql, filter_params)
                        semantic_jobs = [dict(row) for row in cursor.fetchall()]
            except Exception as se:
                logging.warning(f"Semantic search failed, falling back to keyword only: {se}")

        # ---- 3. Merge & deduplicate ----
        all_jobs_map: Dict[int, dict] = {}
        for job in keyword_jobs:
            all_jobs_map[job['id']] = job
        for job in semantic_jobs:
            if job['id'] not in all_jobs_map:
                all_jobs_map[job['id']] = job

        # If no query, just list all jobs with filters
        if not query:
            sql = "SELECT * FROM job_store WHERE 1=1"
            params: list = []
            sql, params = _apply_filters(sql, params, filters)
            sql += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, params)
                all_jobs_list = [dict(row) for row in cursor.fetchall()]
            all_jobs_map = {j['id']: j for j in all_jobs_list}

        # ---- 4. Score & rank ----
        results = []
        for jid, job in all_jobs_map.items():
            job_skills = [s.strip() for s in job.get('skills', '').split(';') if s.strip()]
            if len(job_skills) <= 1 and ',' in job.get('skills', ''):
                job_skills = [s.strip() for s in job.get('skills', '').split(',') if s.strip()]

            # Combined scoring
            keyword_hit = 1.0 if jid in keyword_ids else 0.0
            semantic_score = semantic_scores.get(jid, 0.0)
            hybrid_score = 0.4 * keyword_hit + 0.6 * semantic_score if query else 0.0

            match_score = None
            why_match = None

            if cv_info:
                cv_skills = cv_info.get('skills', [])
                matched_skills = set(cv_skills) & set(job_skills)
                skill_score = len(matched_skills) / max(len(job_skills), 1) if job_skills else 0.0
                match_score = 0.5 * skill_score + 0.5 * hybrid_score if query else skill_score
                if matched_skills:
                    why_match = f"Khop {len(matched_skills)} ky nang: {', '.join(list(matched_skills)[:3])}"
                else:
                    why_match = "Co the phu hop voi vi tri nay"
            elif query:
                match_score = hybrid_score

            company_name = job.get('name') or job.get('company_name') or job.get('company') or 'Unknown Company'
            results.append(JobSearchResult(
                job_id=job['id'],
                job_title=job['job_title'],
                company_name=company_name,
                match_score=match_score,
                salary=job.get('salary', 'N/A'),
                work_location=job.get('work_location', 'N/A'),
                work_type=job.get('work_type', 'N/A'),
                deadline=job.get('deadline', 'N/A'),
                why_match=why_match
            ))

        # Sort by match_score descending
        results.sort(key=lambda x: x.match_score or 0, reverse=True)

        # Apply pagination on merged results
        total = len(results)
        results = results[offset:offset + limit]

        logging.info(f"Hybrid search: {total} total, returning {len(results)} jobs "
                     f"(keyword={len(keyword_ids)}, semantic={len(semantic_scores)})")
        return JobSearchResponse(
            total=total,
            jobs=results,
            limit=limit,
            offset=offset
        )
    except Exception as e:
        logging.error(f"Loi tim kiem jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Loi tim kiem: {str(e)}")


def _build_filter_conditions(filters: Dict) -> tuple:
    """Build SQL filter conditions and params from filters dict."""
    conditions: List[str] = []
    params: list = []
    if filters.get('work_location'):
        locations = filters['work_location']
        placeholders = ','.join(['?' for _ in locations])
        conditions.append(f"work_location IN ({placeholders})")
        params.extend(locations)
    if filters.get('work_type'):
        work_types = filters['work_type']
        placeholders = ','.join(['?' for _ in work_types])
        conditions.append(f"work_type IN ({placeholders})")
        params.extend(work_types)
    if filters.get('experience'):
        conditions.append("experience = ?")
        params.append(filters['experience'])
    return conditions, params


def _apply_filters(sql: str, params: list, filters: Dict) -> tuple:
    """Apply filter conditions to SQL query."""
    conditions, filter_params = _build_filter_conditions(filters)
    for cond in conditions:
        sql += f" AND {cond}"
    params.extend(filter_params)
    return sql, params

@router.get("/analytics")
async def get_jobs_analytics():
    """
    Phân tích xu hướng việc làm - Dashboard Analytics
    Returns:
    - top_job_titles: Top 10 vị trí tuyển dụng nhiều nhất
    - top_companies: Top 10 công ty tuyển dụng nhiều nhất
    - salary_distribution: Phân bố mức lương
    - location_distribution: Phân bố địa điểm làm việc
    - job_type_distribution: Phân bố loại hình công việc
    - experience_distribution: Phân bố yêu cầu kinh nghiệm
    - top_skills: Top 20 kỹ năng được yêu cầu nhiều nhất
    - deadline_stats: Thống kê deadline (sắp hết hạn, còn lâu)
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # 1. Top 10 Job Titles
            cursor.execute("""
                SELECT job_title, COUNT(*) as count
                FROM job_store
                WHERE job_title IS NOT NULL AND job_title != ''
                GROUP BY job_title
                ORDER BY count DESC
                LIMIT 10
            """)
            top_job_titles = [{"title": row["job_title"], "count": row["count"]} for row in cursor.fetchall()]
            # 2. Top 10 Companies
            cursor.execute("""
                SELECT name, COUNT(*) as count
                FROM job_store
                WHERE name IS NOT NULL AND name != ''
                GROUP BY name
                ORDER BY count DESC
                LIMIT 10
            """)
            top_companies = [{"company": row["name"], "count": row["count"]} for row in cursor.fetchall()]
            # 3. Salary Distribution (phân loại theo range)
            cursor.execute("""
                SELECT salary, COUNT(*) as count
                FROM job_store
                WHERE salary IS NOT NULL AND salary != '' AND salary != 'Thỏa thuận'
                GROUP BY salary
                ORDER BY count DESC
                LIMIT 15
            """)
            salary_distribution = [{"salary": row["salary"], "count": row["count"]} for row in cursor.fetchall()]
            # 4. Location Distribution
            cursor.execute("""
                SELECT work_location, COUNT(*) as count
                FROM job_store
                WHERE work_location IS NOT NULL AND work_location != ''
                GROUP BY work_location
                ORDER BY count DESC
                LIMIT 10
            """)
            location_distribution = [{"location": row["work_location"], "count": row["count"]} for row in cursor.fetchall()]
            # 5. Job Type Distribution (work_type column)
            cursor.execute("""
                SELECT work_type, COUNT(*) as count
                FROM job_store
                WHERE work_type IS NOT NULL AND work_type != ''
                GROUP BY work_type
                ORDER BY count DESC
            """)
            job_type_distribution = [{"type": row["work_type"], "count": row["count"]} for row in cursor.fetchall()]
            # 6. Experience Distribution
            cursor.execute("""
                SELECT experience, COUNT(*) as count
                FROM job_store
                WHERE experience IS NOT NULL AND experience != ''
                GROUP BY experience
                ORDER BY count DESC
            """)
            experience_distribution = [{"experience": row["experience"], "count": row["count"]} for row in cursor.fetchall()]
            # 7. Top Skills (parse từ skills JSON array)
            cursor.execute("SELECT skills FROM job_store WHERE skills IS NOT NULL AND skills != ''")
            skills_counter = {}
            for row in cursor.fetchall():
                try:
                    skills_list = json.loads(row["skills"]) if isinstance(row["skills"], str) else row["skills"]
                    if isinstance(skills_list, list):
                        for skill in skills_list:
                            if skill and isinstance(skill, str):
                                skill = skill.strip()
                                skills_counter[skill] = skills_counter.get(skill, 0) + 1
                except:
                    pass
            top_skills = [{"skill": skill, "count": count} for skill, count in sorted(skills_counter.items(), key=lambda x: x[1], reverse=True)[:20]]
            # 8. Deadline Stats (sắp hết hạn trong 7 ngày, 30 ngày)
            today = datetime.now().date()
            deadline_7_days = (today + timedelta(days=7)).isoformat()
            deadline_30_days = (today + timedelta(days=30)).isoformat()
            cursor.execute("""
                SELECT
                    COUNT(CASE WHEN deadline <= ? THEN 1 END) as expiring_7_days,
                    COUNT(CASE WHEN deadline <= ? THEN 1 END) as expiring_30_days,
                    COUNT(*) as total
                FROM job_store
                WHERE deadline IS NOT NULL AND deadline != ''
            """, (deadline_7_days, deadline_30_days))
            deadline_row = cursor.fetchone()
            deadline_stats = {
                "expiring_7_days": deadline_row["expiring_7_days"] or 0,
                "expiring_30_days": deadline_row["expiring_30_days"] or 0,
                "total_with_deadline": deadline_row["total"] or 0
            }
            # 9. Total Stats
            cursor.execute("SELECT COUNT(*) as total FROM job_store")
            total_jobs = cursor.fetchone()["total"]
            logging.info(f" Phân tích {total_jobs} jobs thành công")
            return {
                "total_jobs": total_jobs,
                "top_job_titles": top_job_titles,
                "top_companies": top_companies,
                "salary_distribution": salary_distribution,
                "location_distribution": location_distribution,
                "job_type_distribution": job_type_distribution,
                "experience_distribution": experience_distribution,
                "top_skills": top_skills,
                "deadline_stats": deadline_stats
            }
    except Exception as e:
        logging.error(f" Lỗi phân tích jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi phân tích: {str(e)}")


@router.post("/analytics/insights")
async def generate_chart_insights(request: Dict[str, Any]):
    """
    Generate AI insights for dashboard charts using LLM
    Request body:
    {
        "chart_type": "top_jobs" | "top_companies" | "location" | "job_type" | "experience" | "salary",
        "data": [...chart data...]
    }
    Returns:
    {
        "analysis": "AI-generated analysis text in Vietnamese"
    }
    """
    try:
        chart_type = request.get("chart_type")
        data = request.get("data", [])
        if not chart_type or not data:
            return {"analysis": "Thiếu thông tin biểu đồ để phân tích."}
        
        # Import LLM from ai_analysis
        from services.ai_analysis import get_llm
        llm_instance = get_llm()
        
        # Import chart prompts
        from prompts import chart_insights_prompts
        
        # Get prompt for chart_type
        prompt_template = chart_insights_prompts.get(chart_type)
        if not prompt_template:
            return {"analysis": "Loại biểu đồ không hợp lệ."}
        
        # Format data_str
        if chart_type == "top_jobs":
            data_str = "\n".join([f"- {item.get('title', 'N/A')}: {item.get('count', 0)} việc làm" for item in data[:10]])
        elif chart_type == "top_companies":
            data_str = "\n".join([f"- {item.get('company', 'N/A')}: {item.get('count', 0)} việc làm" for item in data[:10]])
        elif chart_type == "location":
            data_str = "\n".join([f"- {item.get('location', 'N/A')}: {item.get('count', 0)} việc làm" for item in data[:10]])
        elif chart_type == "job_type":
            data_str = "\n".join([f"- {item.get('type', 'N/A')}: {item.get('count', 0)} việc làm" for item in data])
        elif chart_type == "experience":
            data_str = "\n".join([f"- {item.get('experience', 'N/A')}: {item.get('count', 0)} việc làm" for item in data[:10]])
        elif chart_type == "salary":
            data_str = "\n".join([f"- {item.get('salary', 'N/A')}: {item.get('count', 0)} việc làm" for item in data[:10]])
        else:
            data_str = ""
        
        # Invoke prompt with data_str
        formatted_prompt = prompt_template.format(data_str=data_str)
        logging.info(f"Generating analysis for chart type: {chart_type}")
        response = await llm_instance.ainvoke(formatted_prompt)
        analysis = response.content.strip() # type: ignore
        logging.info(f"Generated analysis: {analysis[:100]}...")
        return {"analysis": analysis}
    except Exception as e:
        logging.error(f"Lỗi generate chart insights: {e}")
        return {"analysis": "Không thể tạo phân tích. Vui lòng thử lại sau."}


@router.post("/reindex")
async def reindex_jobs_to_chroma():
    """
    Force re-index tat ca jobs tu PostgreSQL vao ChromaDB.
    Xoa collection cu va tao lai tu dau.
    """
    try:
        from services.chroma_utils import preload_jobs_from_pg
        
        logging.info(" Starting force re-index from PostgreSQL to ChromaDB...")
        
        success = preload_jobs_from_pg(batch_size=50, force=True)
        
        if success:
            from services.chroma_utils import get_vectorstore
            count = get_vectorstore("jobs")._collection.count()
            return {
                "success": True,
                "message": f"Da index {count} jobs tu PostgreSQL vao ChromaDB",
                "total": count
            }
        else:
            return {
                "success": False,
                "message": "Khong the index jobs tu PostgreSQL",
                "total": 0
            }
        
    except Exception as e:
        logging.exception(f" Error re-indexing jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Loi re-index jobs: {str(e)}")
