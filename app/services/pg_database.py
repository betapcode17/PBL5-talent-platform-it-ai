# app/services/pg_database.py
"""
PostgreSQL database service for querying job data from the Prisma-managed it_job_db.

Actual table/column names (snake_case):
  JobPost: job_post_id, employee_id, company_id, job_type_id, category_id,
           name, job_title, job_url, job_description, candidate_requirements,
           benefits, work_location, work_time, work_type (enum), level,
           experience, education, salary, number_of_hires, deadline,
           created_date, updated_date, is_active
  Company: company_id, company_name, profile_description, company_type,
           company_industry, company_size, country, city,
           company_website_url, company_email, company_image, ...
  Skill:   skill_id, skill_name, skill_type
  JobPostSkill: job_post_id, skill_id, required_level, is_mandatory, priority
  Category: category_id, name
  JobType:  job_type_id, job_type
"""

import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Any

import psycopg2
import psycopg2.extras

from config import DATABASE_URL

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

@contextmanager
def get_pg_connection():
    """Get a PostgreSQL connection (context-managed)."""
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        conn.close()


def test_connection() -> bool:
    """Test PostgreSQL connectivity."""
    try:
        with get_pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                logger.info("✅ PostgreSQL connection OK")
                return True
    except Exception as e:
        logger.error(f"❌ PostgreSQL connection failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Job queries
# ---------------------------------------------------------------------------

def get_all_jobs(limit: int = 5000) -> List[Dict[str, Any]]:
    """
    Fetch all active jobs with company, category, jobType joined.
    Returns list of dicts ready for ChromaDB indexing.
    """
    query = """
        SELECT
            jp.job_post_id    AS job_id,
            jp.name           AS company_short_name,
            jp.job_title,
            jp.job_url,
            jp.job_description,
            jp.candidate_requirements,
            jp.benefits,
            jp.work_location,
            jp.work_time,
            jp.work_type::text AS work_type,
            jp.level,
            jp.experience,
            jp.education,
            jp.salary,
            jp.number_of_hires,
            jp.deadline,
            jp.created_date,
            -- category
            cat.name          AS category_name,
            -- job type
            jt.job_type       AS job_type_name,
            -- company info
            c.company_name,
            c.company_image,
            c.profile_description AS company_description,
            c.company_size,
            c.company_industry,
            c.city            AS company_city,
            c.country         AS company_country,
            c.company_website_url
        FROM "JobPost" jp
        LEFT JOIN "Company"   c   ON jp.company_id   = c.company_id
        LEFT JOIN "Category"  cat ON jp.category_id   = cat.category_id
        LEFT JOIN "JobType"   jt  ON jp.job_type_id   = jt.job_type_id
        WHERE jp.is_active = true
        ORDER BY jp.created_date DESC
        LIMIT %s
    """

    with get_pg_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, (limit,))
            rows = cur.fetchall()

    # Attach skills for each job
    job_ids = [r["job_id"] for r in rows]
    skills_map = _get_skills_for_jobs(job_ids) if job_ids else {}

    results = []
    for row in rows:
        job = dict(row)
        jid = job["job_id"]
        job["skills"] = skills_map.get(jid, [])
        job["skills_text"] = ", ".join(job["skills"])
        results.append(job)

    logger.info(f"📦 Fetched {len(results)} jobs from PostgreSQL")
    return results


def _get_skills_for_jobs(job_ids: List[int]) -> Dict[int, List[str]]:
    """Batch-fetch skills for a list of job IDs."""
    if not job_ids:
        return {}

    query = """
        SELECT jps.job_post_id, s.skill_name
        FROM "JobPostSkill" jps
        JOIN "Skill" s ON jps.skill_id = s.skill_id
        WHERE jps.job_post_id = ANY(%s)
    """
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (job_ids,))
            rows = cur.fetchall()

    mapping: Dict[int, List[str]] = {}
    for job_id, skill_name in rows:
        mapping.setdefault(job_id, []).append(skill_name)
    return mapping


def get_job_by_id(job_id: int) -> Optional[Dict[str, Any]]:
    """Fetch a single job post by ID."""
    query = """
        SELECT
            jp.job_post_id AS job_id, jp.job_title, jp.job_description,
            jp.candidate_requirements, jp.benefits, jp.salary,
            jp.experience, jp.education,
            jp.number_of_hires, jp.deadline, jp.work_location,
            jp.work_type::text AS work_type, jp.level,
            jt.job_type       AS job_type_name,
            cat.name          AS category_name,
            c.company_name, c.company_image,
            c.city AS company_city, c.company_website_url
        FROM "JobPost" jp
        LEFT JOIN "Company"  c   ON jp.company_id   = c.company_id
        LEFT JOIN "Category" cat ON jp.category_id   = cat.category_id
        LEFT JOIN "JobType"  jt  ON jp.job_type_id   = jt.job_type_id
        WHERE jp.job_post_id = %s
    """
    with get_pg_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, (job_id,))
            row = cur.fetchone()

    if not row:
        return None

    job = dict(row)
    skills_map = _get_skills_for_jobs([job_id])
    job["skills"] = skills_map.get(job_id, [])
    job["skills_text"] = ", ".join(job["skills"])
    return job


def search_jobs_by_keyword(keyword: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Full-text keyword search on job_title and job_description."""
    query = """
        SELECT
            jp.job_post_id AS job_id, jp.job_title, jp.job_description,
            jp.candidate_requirements, jp.salary, jp.work_location,
            jp.experience,
            c.company_name,
            cat.name AS category_name,
            jp.work_type::text AS work_type
        FROM "JobPost" jp
        LEFT JOIN "Company"  c   ON jp.company_id   = c.company_id
        LEFT JOIN "Category" cat ON jp.category_id   = cat.category_id
        WHERE jp.is_active = true
          AND (jp.job_title ILIKE %s OR jp.job_description ILIKE %s)
        ORDER BY jp.created_date DESC
        LIMIT %s
    """
    pattern = f"%{keyword}%"
    with get_pg_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, (pattern, pattern, limit))
            rows = cur.fetchall()

    job_ids = [r["job_id"] for r in rows]
    skills_map = _get_skills_for_jobs(job_ids) if job_ids else {}

    results = []
    for row in rows:
        job = dict(row)
        jid = job["job_id"]
        job["skills"] = skills_map.get(jid, [])
        job["skills_text"] = ", ".join(job["skills"])
        results.append(job)

    return results


# ---------------------------------------------------------------------------
# Statistics / aggregate queries (for chat insights)
# ---------------------------------------------------------------------------

def get_job_stats() -> Dict[str, Any]:
    """Get aggregate statistics about jobs for chatbot context."""
    stats: Dict[str, Any] = {}
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT COUNT(*) FROM "JobPost" WHERE is_active = true')
            stats["total_jobs"] = cur.fetchone()[0] # type: ignore

            cur.execute('SELECT COUNT(DISTINCT company_id) FROM "JobPost" WHERE is_active = true')
            stats["total_companies"] = cur.fetchone()[0] # type: ignore

            # Top categories
            cur.execute("""
                SELECT cat.name, COUNT(*) AS cnt
                FROM "JobPost" jp
                JOIN "Category" cat ON jp.category_id = cat.category_id
                WHERE jp.is_active = true
                GROUP BY cat.name
                ORDER BY cnt DESC
                LIMIT 10
            """)
            stats["top_categories"] = [
                {"name": r[0], "count": r[1]} for r in cur.fetchall()
            ]

            # Top skills
            cur.execute("""
                SELECT s.skill_name, COUNT(*) AS cnt
                FROM "JobPostSkill" jps
                JOIN "Skill" s ON jps.skill_id = s.skill_id
                JOIN "JobPost" jp ON jps.job_post_id = jp.job_post_id
                WHERE jp.is_active = true
                GROUP BY s.skill_name
                ORDER BY cnt DESC
                LIMIT 10
            """)
            stats["top_skills"] = [
                {"name": r[0], "count": r[1]} for r in cur.fetchall()
            ]

    logger.info(f"📊 Job stats: {stats['total_jobs']} active jobs, {stats['total_companies']} companies")
    return stats
