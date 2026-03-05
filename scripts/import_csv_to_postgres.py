"""
Import job data from CSV files into PostgreSQL database (it_job_db).
Reads jobs_vietnamese.csv and job_descriptions.csv,
inserts into Company, Employee, Category, JobType, Skill, JobPost, JobPostSkill.

Usage:
    cd e:\\HOCKI6\\PBL5\\AI
    python scripts/import_csv_to_postgres.py
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import psycopg2
import psycopg2.extras

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:17122005@localhost:5432/it_job_db",
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "app" / "data"

CSV_VIETNAMESE = DATA_DIR / "jobs_vietnamese.csv"
CSV_ITVIEC = DATA_DIR / "job_descriptions.csv"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# Cache dicts to avoid repeated SELECTs
_company_cache = {}   # company_name -> company_id
_category_cache = {}  # name -> category_id
_jobtype_cache = {}   # job_type -> job_type_id
_skill_cache = {}     # skill_name -> skill_id
_employee_cache = {}  # company_id -> employee_id


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

DEFAULT_EMPLOYEE_ID = 6  # HR Manager (company_id=1), always exists


def ensure_employee(cur, company_id):
    """Return an employee_id for the given company. Fallback to DEFAULT_EMPLOYEE_ID."""
    if company_id in _employee_cache:
        return _employee_cache[company_id]

    cur.execute(
        'SELECT employee_id FROM "Employee" WHERE company_id = %s LIMIT 1',
        (company_id,),
    )
    row = cur.fetchone()
    if row:
        _employee_cache[company_id] = row[0]
        return row[0]

    # No employee for this company — use default
    _employee_cache[company_id] = DEFAULT_EMPLOYEE_ID
    return DEFAULT_EMPLOYEE_ID


def upsert_company(cur, name, **kw):
    if not name or not name.strip():
        return None
    name = name.strip()
    if name in _company_cache:
        return _company_cache[name]

    cur.execute('SELECT company_id FROM "Company" WHERE company_name = %s', (name,))
    row = cur.fetchone()
    if row:
        _company_cache[name] = row[0]
        return row[0]

    cur.execute(
        """INSERT INTO "Company" (
              company_name, company_image, company_size,
              company_industry, company_website_url, company_email,
              is_active, created_date
           ) VALUES (%s,%s,%s,%s,%s,%s,true,NOW())
           RETURNING company_id""",
        (
            name,
            kw.get("company_image") or "",
            kw.get("company_size") or "",
            kw.get("company_industry") or "",
            kw.get("company_website_url") or "",
            "",
        ),
    )
    cid = cur.fetchone()[0]
    _company_cache[name] = cid
    return cid


def upsert_category(cur, name):
    if not name or not name.strip() or name in ("Không xác định", "nan"):
        return None
    name = name.strip()
    if name in _category_cache:
        return _category_cache[name]

    cur.execute('SELECT category_id FROM "Category" WHERE name = %s', (name,))
    row = cur.fetchone()
    if row:
        _category_cache[name] = row[0]
        return row[0]

    cur.execute(
        'INSERT INTO "Category" (name) VALUES (%s) RETURNING category_id', (name,)
    )
    cid = cur.fetchone()[0]
    _category_cache[name] = cid
    return cid


def upsert_job_type(cur, text):
    if not text or not text.strip() or text in ("Không xác định", "nan"):
        return None
    text = text.strip()
    if text in _jobtype_cache:
        return _jobtype_cache[text]

    cur.execute('SELECT job_type_id FROM "JobType" WHERE job_type = %s', (text,))
    row = cur.fetchone()
    if row:
        _jobtype_cache[text] = row[0]
        return row[0]

    cur.execute(
        'INSERT INTO "JobType" (job_type) VALUES (%s) RETURNING job_type_id', (text,)
    )
    jid = cur.fetchone()[0]
    _jobtype_cache[text] = jid
    return jid


def upsert_skill(cur, skill_name):
    if not skill_name or not skill_name.strip():
        return None
    skill_name = skill_name.strip()
    if skill_name in _skill_cache:
        return _skill_cache[skill_name]

    cur.execute('SELECT skill_id FROM "Skill" WHERE skill_name = %s', (skill_name,))
    row = cur.fetchone()
    if row:
        _skill_cache[skill_name] = row[0]
        return row[0]

    cur.execute(
        """INSERT INTO "Skill" (skill_name, skill_type, created_date, updated_date)
           VALUES (%s, 'TECHNICAL', NOW(), NOW()) RETURNING skill_id""",
        (skill_name,),
    )
    sid = cur.fetchone()[0]
    _skill_cache[skill_name] = sid
    return sid


def map_work_type(text):
    """Map Vietnamese work-type text → WorkType enum (ONSITE | REMOTE | HYBRID)."""
    if not text:
        return "ONSITE"
    t = text.strip().lower()
    if any(k in t for k in ("remote", "từ xa")):
        return "REMOTE"
    if any(k in t for k in ("hybrid", "kết hợp")):
        return "HYBRID"
    return "ONSITE"   # Toàn thời gian, bán thời gian, etc. → ONSITE


def parse_deadline(text):
    if not text or str(text) in ("nan", "Không xác định", ""):
        return None
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(str(text).strip(), fmt)
        except ValueError:
            continue
    return None


def safe_int(val, default=1):
    try:
        v = int(float(val))
        return v if v > 0 else default
    except (ValueError, TypeError):
        return default


def safe_str(val):
    s = str(val) if val is not None else ""
    return "" if s == "nan" else s


# -------------------------------------------------------------------
# Default fallback IDs (created once)
# -------------------------------------------------------------------
_default_category_id = None
_default_jobtype_id = None


def get_default_category(cur):
    global _default_category_id
    if _default_category_id:
        return _default_category_id
    _default_category_id = upsert_category(cur, "Khác")
    if not _default_category_id:
        cur.execute('INSERT INTO "Category" (name) VALUES (%s) RETURNING category_id', ("Khác",))
        _default_category_id = cur.fetchone()[0]
    return _default_category_id


def get_default_jobtype(cur):
    global _default_jobtype_id
    if _default_jobtype_id:
        return _default_jobtype_id
    _default_jobtype_id = upsert_job_type(cur, "Toàn thời gian")
    if not _default_jobtype_id:
        cur.execute('INSERT INTO "JobType" (job_type) VALUES (%s) RETURNING job_type_id', ("Toàn thời gian",))
        _default_jobtype_id = cur.fetchone()[0]
    return _default_jobtype_id


# -------------------------------------------------------------------
# Import: jobs_vietnamese.csv
# -------------------------------------------------------------------

def import_vietnamese_csv(conn):
    if not CSV_VIETNAMESE.exists():
        log.warning(f"File not found: {CSV_VIETNAMESE}")
        return 0

    df = pd.read_csv(CSV_VIETNAMESE, encoding="utf-8-sig")
    log.info(f"📄 Loaded {len(df)} rows from jobs_vietnamese.csv")

    inserted = skipped = errors = 0
    cur = conn.cursor()

    for idx, row in df.iterrows():
        try:
            job_title = safe_str(row.get("Chức danh công việc")).strip()
            if not job_title:
                skipped += 1
                continue

            job_url = safe_str(row.get("Đường dẫn công việc")).strip()
            if job_url:
                cur.execute('SELECT 1 FROM "JobPost" WHERE job_url = %s', (job_url,))
                if cur.fetchone():
                    skipped += 1
                    continue

            # Company
            company_name = safe_str(row.get("Tên công ty")).strip()
            company_id = upsert_company(
                cur,
                company_name,
                company_image=safe_str(row.get("Logo công ty")),
                company_size=safe_str(row.get("Quy mô công ty")),
                company_industry=safe_str(row.get("Lĩnh vực công ty")),
                company_website_url=safe_str(row.get("Website công ty")),
            )
            if not company_id:
                # Create a generic company
                company_id = upsert_company(cur, "Unknown Company")

            employee_id = ensure_employee(cur, company_id)

            # Category (first item from semicolon list)
            cats_raw = safe_str(row.get("Danh mục liên quan"))
            first_cat = cats_raw.split(";")[0].strip() if cats_raw else ""
            category_id = upsert_category(cur, first_cat) or get_default_category(cur)

            # JobType
            wt_text = safe_str(row.get("Hình thức làm việc")).strip()
            job_type_id = upsert_job_type(cur, wt_text) or get_default_jobtype(cur)

            work_type_enum = map_work_type(wt_text)
            deadline = parse_deadline(row.get("Hạn nộp hồ sơ"))
            num_hires = safe_int(row.get("Số lượng tuyển"), 1)

            cur.execute(
                """
                INSERT INTO "JobPost" (
                    employee_id, company_id, job_type_id, category_id,
                    name, job_title, job_url,
                    job_description, candidate_requirements, benefits,
                    work_location, work_time, work_type,
                    level, experience, education,
                    salary, number_of_hires, deadline,
                    created_date, updated_date, is_active
                ) VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s::"WorkType",
                    %s, %s, %s,
                    %s, %s, %s,
                    NOW(), NOW(), true
                ) RETURNING job_post_id
                """,
                (
                    employee_id, company_id, job_type_id, category_id,
                    company_name, job_title, job_url or None,
                    safe_str(row.get("Mô tả công việc")),
                    safe_str(row.get("Yêu cầu ứng viên")),
                    safe_str(row.get("Quyền lợi")),
                    safe_str(row.get("Địa điểm làm việc")),
                    safe_str(row.get("Thời gian làm việc")),
                    work_type_enum,
                    safe_str(row.get("Cấp bậc")),
                    safe_str(row.get("Kinh nghiệm")),
                    safe_str(row.get("Trình độ học vấn")),
                    safe_str(row.get("Mức lương")),
                    num_hires,
                    deadline,
                ),
            )
            job_post_id = cur.fetchone()[0]

            # Skills (semicolon-separated)
            skills_raw = safe_str(row.get("Kỹ năng"))
            if skills_raw and skills_raw not in ("Không xác định",):
                for sname in skills_raw.split(";"):
                    sname = sname.strip()
                    if not sname:
                        continue
                    sid = upsert_skill(cur, sname)
                    if sid:
                        cur.execute(
                            """INSERT INTO "JobPostSkill" (job_post_id, skill_id, is_mandatory, priority)
                               VALUES (%s, %s, true, 1)
                               ON CONFLICT (job_post_id, skill_id) DO NOTHING""",
                            (job_post_id, sid),
                        )

            inserted += 1
            if inserted % 200 == 0:
                conn.commit()
                log.info(f"  ... {inserted} jobs inserted")

        except Exception as e:
            errors += 1
            conn.rollback()
            if errors <= 5:
                log.warning(f"⚠️ Row {idx} error: {e}")

    conn.commit()
    log.info(
        f"✅ jobs_vietnamese.csv: inserted={inserted}, skipped={skipped}, errors={errors}"
    )
    return inserted


# -------------------------------------------------------------------
# Import: job_descriptions.csv (ITViec)
# -------------------------------------------------------------------

def import_itviec_csv(conn):
    if not CSV_ITVIEC.exists():
        log.warning(f"File not found: {CSV_ITVIEC}")
        return 0

    df = pd.read_csv(CSV_ITVIEC, encoding="utf-8-sig")
    log.info(f"📄 Loaded {len(df)} rows from job_descriptions.csv")

    inserted = skipped = errors = 0
    cur = conn.cursor()

    for idx, row in df.iterrows():
        try:
            job_title = safe_str(row.get("title")).strip()
            if not job_title:
                skipped += 1
                continue

            job_url = safe_str(row.get("job_url")).strip()
            if job_url:
                cur.execute('SELECT 1 FROM "JobPost" WHERE job_url = %s', (job_url,))
                if cur.fetchone():
                    skipped += 1
                    continue

            # Company
            company_name = safe_str(row.get("company")).strip()
            company_id = upsert_company(
                cur,
                company_name,
                company_image=safe_str(row.get("company_image_url")),
            )
            if not company_id:
                company_id = upsert_company(cur, "Unknown Company")

            employee_id = ensure_employee(cur, company_id)

            # Category from it_role_type
            it_role = safe_str(row.get("it_role_type")).strip()
            category_id = upsert_category(cur, it_role) or get_default_category(cur)

            job_type_id = get_default_jobtype(cur)

            location = safe_str(row.get("location")).strip()
            city = safe_str(row.get("city")).strip()
            work_location = location or city
            description = safe_str(row.get("description"))

            cur.execute(
                """
                INSERT INTO "JobPost" (
                    employee_id, company_id, job_type_id, category_id,
                    name, job_title, job_url,
                    job_description, candidate_requirements, benefits,
                    work_location, work_time, work_type,
                    level, experience, education,
                    salary, number_of_hires, deadline,
                    created_date, updated_date, is_active
                ) VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s::"WorkType",
                    %s, %s, %s,
                    %s, %s, %s,
                    NOW(), NOW(), true
                ) RETURNING job_post_id
                """,
                (
                    employee_id, company_id, job_type_id, category_id,
                    company_name, job_title, job_url or None,
                    description, "", "",
                    work_location, "", "ONSITE",
                    "", "", "",
                    "", 1, None,
                ),
            )
            job_post_id = cur.fetchone()[0]

            # Skills from main_programming_languages + key_technologies
            skills_set = set()
            for col in ("main_programming_languages", "key_technologies"):
                raw = safe_str(row.get(col))
                for s in raw.split(","):
                    s = s.strip()
                    if s:
                        skills_set.add(s)

            for sname in skills_set:
                sid = upsert_skill(cur, sname)
                if sid:
                    cur.execute(
                        """INSERT INTO "JobPostSkill" (job_post_id, skill_id, is_mandatory, priority)
                           VALUES (%s, %s, true, 1)
                           ON CONFLICT (job_post_id, skill_id) DO NOTHING""",
                        (job_post_id, sid),
                    )

            inserted += 1
            if inserted % 200 == 0:
                conn.commit()
                log.info(f"  ... {inserted} jobs inserted")

        except Exception as e:
            errors += 1
            conn.rollback()
            if errors <= 5:
                log.warning(f"⚠️ Row {idx} error: {e}")

    conn.commit()
    log.info(
        f"✅ job_descriptions.csv: inserted={inserted}, skipped={skipped}, errors={errors}"
    )
    return inserted


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    log.info("🚀 Starting CSV → PostgreSQL import")
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False

    try:
        count1 = import_vietnamese_csv(conn)
        count2 = import_itviec_csv(conn)

        # Summary
        cur = conn.cursor()
        for tbl in ("JobPost", "Company", "Skill", "Category", "JobType", "Employee", "JobPostSkill"):
            cur.execute(f'SELECT COUNT(*) FROM "{tbl}"')
            log.info(f"   {tbl:20s}: {cur.fetchone()[0]}") # type: ignore
        cur.close()

        log.info("=" * 50)
        log.info(f"📊 DONE — Inserted {count1} (VN) + {count2} (ITViec) = {count1+count2} jobs")
        log.info("=" * 50)

    except Exception as e:
        conn.rollback()
        log.error(f"❌ Import failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
