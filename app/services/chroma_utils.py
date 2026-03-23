# app/services/chroma_utils.py
"""
Chroma utilities for vector storage and job/CV preloading.
Supports separate collections for jobs and CVs (for reverse matching).
"""

import os
import json
import logging
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import pandas as pd  # Fixed: import pandas as pd (not from turtle)

from .api_key_manager import get_next_api_key
from .db_utils import get_db_connection, create_tables
from .pg_database import get_all_jobs

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global instances for job and CV collections
_job_vectorstore = None
_cv_vectorstore = None

def get_vectorstore(collection_name: str = "jobs") -> Chroma:
    """
    Khởi tạo Chroma vectorstore với Google Gemini Embedding API.
    Separate collections: "jobs" (default), "cvs" for reverse matching.
    Model: text-embedding-004 (miễn phí, hỗ trợ multilingual).
    """
    global _job_vectorstore, _cv_vectorstore
    
    if collection_name == "cvs":
        if _cv_vectorstore is None:
            _cv_vectorstore = _initialize_vectorstore("cvs")
        return _cv_vectorstore
    else:  # "jobs" default
        if _job_vectorstore is None:
            _job_vectorstore = _initialize_vectorstore("jobs")
        return _job_vectorstore

def _initialize_vectorstore(collection_name: str) -> Chroma:
    """Internal init for a specific collection."""
    try:
        google_api_key = get_next_api_key()
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        base_dir = Path(__file__).resolve().parent.parent  # app/ -> root
        chroma_path = base_dir / "db" / "chroma_db" / collection_name
        chroma_path.mkdir(parents=True, exist_ok=True)
        
        embedding_function = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=google_api_key, # type: ignore
            task_type="retrieval_document"  # Tối ưu cho retrieval
        )
        
        vectorstore = Chroma(
            persist_directory=str(chroma_path),
            collection_name=collection_name,
            embedding_function=embedding_function
        )
        logging.info(f" Initialized Chroma vectorstore for '{collection_name}' with Google Gemini Embedding API")
        return vectorstore
    except Exception as e:
        logging.error(f" Error initializing Chroma for '{collection_name}': {e}")
        raise

def preload_jobs(csv_path: str, batch_size: int = 1000) -> bool:
    """
    Preload jobs from CSV file into SQLite and Chroma (jobs collection).
    Accepts both str and Path.
    """
    try:
        csv_path = Path(csv_path) # type: ignore

        if not csv_path.exists(): # type: ignore
            logging.error(f" File does not exist: {csv_path}")
            return False

        if csv_path.suffix.lower() != ".csv": # type: ignore
            raise ValueError(f"File must be .csv, got: {csv_path}")

        logging.info(f" Preloading jobs from {csv_path}")

        create_tables()
        logging.info(" Ensured database tables are created")

        vectorstore = get_vectorstore("jobs")  # Use jobs collection

        with get_db_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM job_store")
            if cursor.fetchone()[0] > 0:
                logging.info(" Skipping preload: job_store already populated")
                return True

            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            logging.info(f" Loaded {len(df)} jobs from CSV")

            documents = []
            inserted_count = 0

            for index, row in df.iterrows():
                try:
                    job_data = {
                        "name": str(row.get("Tên công ty", "") or ""),
                        "job_title": str(row.get("Chức danh công việc", "") or ""),
                        "job_url": str(row.get("Đường dẫn công việc", "") or ""),
                        "job_description": str(row.get("Mô tả công việc", "") or ""),
                        "candidate_requirements": str(row.get("Yêu cầu ứng viên", "") or ""),
                        "benefits": str(row.get("Quyền lợi", "") or ""),
                        "work_location": str(row.get("Địa điểm làm việc", "") or ""),
                        "work_time": str(row.get("Thời gian làm việc", "") or ""),
                        "job_tags": str(row.get("Thẻ công việc", "") or ""),
                        "skills": str(row.get("Kỹ năng", "") or ""),
                        "related_categories": str(row.get("Danh mục liên quan", "") or ""),
                        "salary": str(row.get("Mức lương", "") or ""),
                        "experience": str(row.get("Kinh nghiệm", "") or ""),
                        "deadline": str(row.get("Hạn nộp hồ sơ", "") or ""),
                        "company_logo": str(row.get("Logo công ty", "") or ""),
                        "company_scale": str(row.get("Quy mô công ty", "") or ""),
                        "company_field": str(row.get("Lĩnh vực công ty", "") or ""),
                        "company_address": str(row.get("Địa chỉ công ty", "") or ""),
                        "level": str(row.get("Cấp bậc", "") or ""),
                        "education": str(row.get("Trình độ học vấn", "") or ""),
                        "number_of_hires": int(row.get("Số lượng tuyển", 1))
                        if pd.notna(row.get("Số lượng tuyển"))
                        else 1,
                        "work_type": str(row.get("Hình thức làm việc", "") or ""),
                        "company_url": str(row.get("Website công ty", "") or ""),
                        "timestamp": str(row.get("Thời gian lấy dữ liệu", "") or ""),
                    }

                    cursor.execute(
                        """
                        INSERT OR IGNORE INTO job_store (
                            name, job_title, job_url, job_description, candidate_requirements,
                            benefits, work_location, work_time, job_tags, skills,
                            related_categories, salary, experience, deadline, company_logo,
                            company_scale, company_field, company_address, level, education,
                            number_of_hires, work_type, company_url, timestamp
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        tuple(job_data.values()),
                    )

                    if cursor.rowcount > 0:
                        cursor.execute(
                            "SELECT id FROM job_store WHERE job_url = ?",
                            (job_data["job_url"],),
                        )
                        job_data["job_id"] = cursor.fetchone()[0]

                        skills_list = [
                            s.strip()
                            for s in job_data["skills"].split(",")
                            if s.strip()
                        ]

                        page_content = (
                            f"JOB_ID: {job_data['job_id']}\n"  # Prefix for prompt
                            f"{job_data['job_title']} "
                            f"{job_data['job_description']} "
                            f"{job_data['candidate_requirements']} "
                            f"{json.dumps(skills_list, ensure_ascii=False)}"
                        )

                        documents.append(
                            Document(page_content=page_content, metadata=job_data)
                        )
                        inserted_count += 1

                        if len(documents) >= batch_size:
                            vectorstore.add_documents(documents)
                            logging.info(f" Added batch of {len(documents)} to Chroma (jobs collection)")
                            documents.clear()

                except Exception as row_err:
                    # Safe logging for Vietnamese chars
                    safe_err = str(row_err).encode('utf-8', errors='replace').decode('utf-8')
                    logging.warning(f" Skipping row {index}: {safe_err}")

            if documents:
                vectorstore.add_documents(documents)
                logging.info(f" Added final batch of {len(documents)} to Chroma (jobs collection)")

            conn.commit()
            logging.info(f" Preloaded {inserted_count} jobs successfully")

        return True

    except Exception as e:
        safe_e = str(e).encode('utf-8', errors='replace').decode('utf-8')
        logging.error(f" Error preloading jobs from CSV: {safe_e}", exc_info=True)
        return False

async def index_cv_extracts(skills: list, aspirations: str, experience: str, education: str, cv_id: int) -> bool:
    """
    Index CV extracts into Chroma (cvs collection for reverse matching).
    """
    if not isinstance(cv_id, int):
        raise ValueError("cv_id must be an integer")
    try:
        content = (
            f"CV_ID: {cv_id}\n"  # Prefix for prompt (similar to JOB_ID)
            f"Skills: {json.dumps(skills, ensure_ascii=False)} "
            f"Aspirations: {aspirations} "
            f"Experience: {experience} "
            f"Education: {education}"
        )
        doc = Document(page_content=content, metadata={"cv_id": cv_id, "type": "cv"})
        vectorstore = get_vectorstore("cvs")  # Use CV collection
        vectorstore.add_documents([doc])
        logging.info(f" Indexed CV {cv_id} into Chroma (cvs collection)")
        return True
    except Exception as e:
        safe_e = str(e).encode('utf-8', errors='replace').decode('utf-8')
        logging.error(f" Error indexing CV {cv_id}: {safe_e}")
        return False

def delete_cv_from_chroma(cv_id: int) -> bool:
    """
    Delete CV from Chroma (cvs collection).
    """
    if not isinstance(cv_id, int):
        raise ValueError("cv_id must be an integer")
    try:
        vectorstore = get_vectorstore("cvs")
        docs = vectorstore.get(where={"cv_id": cv_id})
        if not docs['ids']:
            logging.info(f"ℹ No documents found for CV {cv_id}")
            return False
        vectorstore._collection.delete(where={"cv_id": cv_id})
        logging.info(f" Deleted CV {cv_id} from Chroma (cvs collection)")
        return True
    except Exception as e:
        safe_e = str(e).encode('utf-8', errors='replace').decode('utf-8')
        logging.error(f" Error deleting CV {cv_id} from Chroma: {safe_e}")
        return False


# ---------------------------------------------------------------------------
# PostgreSQL-based preload
# ---------------------------------------------------------------------------

def preload_jobs_from_pg(batch_size: int = 20, force: bool = False) -> bool:
    """
    Preload jobs from PostgreSQL (it_job_db) into ChromaDB.
    Replaces the CSV-based preload for the RAG chatbot.
    
    Args:
        batch_size: Number of documents per embedding API call
        force: If True, delete existing collection and re-index all jobs
    """
    import time
    try:
        vectorstore = get_vectorstore("jobs")

        # Check if already populated
        existing = vectorstore._collection.count()
        if existing > 0 and not force:
            logging.info(f" Skipping PG preload: jobs collection already has {existing} documents")
            return True

        if force and existing > 0:
            logging.info(f" Force reindex: deleting {existing} existing documents...")
            vectorstore._collection.delete(where={})
            logging.info(" Cleared existing ChromaDB jobs collection")

        logging.info(" Preloading jobs from PostgreSQL into ChromaDB...")
        jobs = get_all_jobs(limit=10000)

        if not jobs:
            logging.warning(" No jobs found in PostgreSQL")
            return False

        documents: List[Document] = []
        for job in jobs:
            skills_list = job.get("skills", [])
            company_name = str(job.get("company_name", "") or job.get("company_short_name", ""))
            page_content = (
                f"JOB_ID: {job['job_id']}\n"
                f"Vi tri: {job.get('job_title', '')} tai {company_name}\n"
                f"Mo ta: {(job.get('job_description', '') or '')[:300]}\n"
                f"Yeu cau: {(job.get('candidate_requirements', '') or '')[:200]}\n"
                f"Ky nang: {json.dumps(skills_list, ensure_ascii=False)}\n"
                f"Dia diem: {job.get('work_location', '')}\n"
                f"Luong: {job.get('salary', '')}\n"
                f"Kinh nghiem: {job.get('experience', '')}\n"
                f"Hinh thuc: {job.get('work_type', '')} {job.get('job_type_name', '')}\n"
                f"Nganh: {job.get('category_name', '')}\n"
                f"Cap bac: {job.get('level', '')}"
            )
            metadata = {
                "job_id": job["job_id"],
                "job_title": str(job.get("job_title", "")),
                "company": company_name,
                "location": str(job.get("work_location", "") or ""),
                "salary": str(job.get("salary", "") or ""),
                "experience": str(job.get("experience", "") or ""),
                "education": str(job.get("education", "") or ""),
                "work_type": str(job.get("work_type", "") or ""),
                "category": str(job.get("category_name", "") or ""),
                "skills": str(job.get("skills_text", "") or ""),
                "url": str(job.get("job_url", "") or ""),
                "num_positions": int(job.get("number_of_hires", 1) or 1),
                # Enriched fields from PostgreSQL
                "benefits": str(job.get("benefits", "") or ""),
                "level": str(job.get("level", "") or ""),
                "deadline": str(job.get("deadline", "") or ""),
                "work_time": str(job.get("work_time", "") or ""),
                "job_type": str(job.get("job_type_name", "") or ""),
                "company_size": str(job.get("company_size", "") or ""),
                "company_industry": str(job.get("company_industry", "") or ""),
                "company_city": str(job.get("company_city", "") or ""),
                "company_website": str(job.get("company_website_url", "") or ""),
                "company_image": str(job.get("company_image", "") or ""),
                "created_date": str(job.get("created_date", "") or ""),
                "job_description": str(job.get("job_description", "") or "")[:500],
                "candidate_requirements": str(job.get("candidate_requirements", "") or "")[:500],
            }
            documents.append(Document(page_content=page_content, metadata=metadata))

        # Add in batches with delay to avoid rate limiting
        total_batches = (len(documents) + batch_size - 1) // batch_size
        for i in range(0, len(documents), batch_size):
            batch = documents[i: i + batch_size]
            batch_num = i // batch_size + 1
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    vectorstore.add_documents(batch)
                    logging.info(f" Added batch {batch_num}/{total_batches} ({len(batch)} docs) to ChromaDB")
                    break
                except Exception as batch_err:
                    if ("429" in str(batch_err) or "RESOURCE_EXHAUSTED" in str(batch_err)):
                        wait_time = 60 * (attempt + 1)
                        logging.warning(f" Rate limited at batch {batch_num} (attempt {attempt+1}), waiting {wait_time}s...")
                        time.sleep(wait_time)
                        if attempt == max_retries - 1:
                            logging.error(f" Failed batch {batch_num} after {max_retries} retries, skipping...")
                    else:
                        raise
            # Delay between batches to respect API rate limits
            if batch_num < total_batches:
                time.sleep(5)

        logging.info(f" Preloaded {len(documents)} jobs from PostgreSQL into ChromaDB")
        return True

    except Exception as e:
        safe_e = str(e).encode('utf-8', errors='replace').decode('utf-8')
        logging.error(f" Error preloading jobs from PostgreSQL: {safe_e}", exc_info=True)
        return False