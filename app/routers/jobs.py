# app/routers/jobs.py
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any
from models.responses import JobSearchInput, JobSearchResponse, JobSearchResult
from services.db_utils import get_db_connection
from services.ai_analysis import get_llm
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
            logging.info(f"✅ Lấy {len(jobs)} jobs (total: {total}, limit: {limit}, offset: {offset})")
            return {
                "jobs": jobs,
                "total": total,
                "limit": limit,
                "offset": offset
            }
    except Exception as e:
        logging.error(f"❌ Lỗi lấy jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi lấy jobs: {str(e)}")

@router.post("/search", response_model=JobSearchResponse)
async def search_jobs_endpoint(input: JobSearchInput):
    """
    Tìm kiếm jobs thông minh với AI ranking
    Khác với /jobs (chỉ list tất cả),
    endpoint này cho phép search với query, filters và AI ranking theo CV.
    """
    try:
        query = input.query
        filters = input.filters
        cv_id = input.cv_id
        limit = input.limit
        offset = input.offset
        logging.info(f"🔍 Tìm kiếm jobs: query='{query}', filters={filters}, cv_id={cv_id}")
        # Lấy CV info nếu có cv_id (để AI ranking)
        cv_info = None
        if cv_id:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT cv_info_json FROM cv_store WHERE id = ?", (cv_id,))
                row = cursor.fetchone()
                if row:
                    cv_info = json.loads(row["cv_info_json"])
        # Build SQL query
        sql = "SELECT * FROM job_store WHERE 1=1"
        params = []
        # Text search
        if query:
            sql += " AND (job_title LIKE ? OR job_description LIKE ? OR skills LIKE ?)"
            search_term = f"%{query}%"
            params.extend([search_term, search_term, search_term])
        # Filters
        if filters.get('work_location'):
            locations = filters['work_location']
            placeholders = ','.join(['?' for _ in locations])
            sql += f" AND work_location IN ({placeholders})"
            params.extend(locations)
        if filters.get('work_type'):
            work_types = filters['work_type']
            placeholders = ','.join(['?' for _ in work_types])
            sql += f" AND work_type IN ({placeholders})"
            params.extend(work_types)
        if filters.get('experience'):
            sql += " AND experience = ?"
            params.append(filters['experience'])
        if filters.get('salary_min'):
            # Simple salary filter (can be improved)
            pass
        # Pagination
        sql += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        # Execute query
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            jobs = [dict(row) for row in cursor.fetchall()]
        # Count total
        count_sql = sql.split("LIMIT")[0]
        count_params = params[:-2]
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) as total FROM ({count_sql})", count_params)
            total = cursor.fetchone()['total']
        # AI ranking nếu có cv_id - SIMPLE MATCHING (không dùng semantic search)
        results = []
        for job in jobs:
            job_skills = [s.strip() for s in job.get('skills', '').split(';') if s.strip()]
            match_score = None
            why_match = None
            if cv_info:
                cv_skills = cv_info.get('skills', [])
                matched_skills = set(cv_skills) & set(job_skills)
                match_score = len(matched_skills) / max(len(job_skills), 1) if job_skills else 0.0
                # Generate simple why_match
                if matched_skills:
                    why_match = f"Khớp {len(matched_skills)} kỹ năng: {', '.join(list(matched_skills)[:3])}"
                else:
                    why_match = "Có thể phù hợp với vị trí này"
            # Get company name from various possible fields
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
        # Sort by match_score if available
        if cv_info:
            results.sort(key=lambda x: x.match_score or 0, reverse=True)
        logging.info(f"✅ Tìm được {total} jobs, trả về {len(results)} jobs")
        return JobSearchResponse(
            total=total,
            jobs=results,
            limit=limit,
            offset=offset
        )
    except Exception as e:
        logging.error(f"❌ Lỗi tìm kiếm jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi tìm kiếm: {str(e)}")

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
            logging.info(f"✅ Phân tích {total_jobs} jobs thành công")
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
        logging.error(f"❌ Lỗi phân tích jobs: {str(e)}")
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
        analysis = response.content.strip()
        logging.info(f"Generated analysis: {analysis[:100]}...")
        return {"analysis": analysis}
    except Exception as e:
        logging.error(f"Lỗi generate chart insights: {e}")
        return {"analysis": "Không thể tạo phân tích. Vui lòng thử lại sau."}