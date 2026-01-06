import json
import logging
import os
from turtle import pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from services.db_utils import create_tables, get_db_connection


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_vectorstore = None

def get_vectorstore():
    """
    Khởi tạo Chroma vectorstore với Google Gemini Embedding API
    Model: text-embedding-004 (miễn phí, hỗ trợ multilingual)
    """
    global _vectorstore
    if _vectorstore is None:
        try:
            # Lấy API key từ environment (use api_key_manager for rotation if needed)
            from services.api_key_manager import get_next_api_key
            google_api_key = get_next_api_key()
            if not google_api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
            base_dir = os.path.dirname(os.path.dirname(__file__))  # Go up to project root
            chroma_path = os.path.join(base_dir, "db", "chroma_db")
            os.makedirs(chroma_path, exist_ok=True)
            # Sử dụng Google Gemini Embedding API
            embedding_function = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=google_api_key,
                task_type="retrieval_document"  # Tối ưu cho retrieval
            )
            _vectorstore = Chroma(
                persist_directory=chroma_path,
                embedding_function=embedding_function
            )
            logging.info("✅ Initialized Chroma vectorstore with Google Gemini Embedding API (text-embedding-004)")
        except Exception as e:
            logging.error(f"❌ Error initializing Chroma vectorstore: {e}")
            raise
    return _vectorstore

from pathlib import Path
import os
import logging
import pandas as pd
import json


def preload_jobs(csv_path, batch_size: int = 1000) -> bool:
    """
    Preload jobs from CSV file into SQLite and Chroma.
    Accepts both str and Path.
    """
    try:
        # ✅ Normalize to Path
        csv_path = Path(csv_path)

        # ✅ Validate file
        if not csv_path.exists():
            logging.error(f"❌ File does not exist: {csv_path}")
            return False

        if csv_path.suffix.lower() != ".csv":
            raise ValueError(f"File must be .csv, got: {csv_path}")

        logging.info(f"📥 Preloading jobs from {csv_path}")

        # Tạo bảng trước khi truy vấn
        create_tables()
        logging.info("✅ Ensured database tables are created")

        vectorstore = get_vectorstore()

        with get_db_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM job_store")
            if cursor.fetchone()[0] > 0:
                logging.info("⏭️ Skipping preload: job_store already populated")
                return True

            # ✅ pandas accepts Path directly
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            logging.info(f"📄 Loaded {len(df)} jobs from CSV")

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
                            logging.info(f"➕ Added batch of {len(documents)} to Chroma")
                            documents.clear()

                except Exception as row_err:
                    logging.warning(f"⚠️ Skipping row {index}: {row_err}")

            if documents:
                vectorstore.add_documents(documents)
                logging.info(f"➕ Added final batch of {len(documents)} to Chroma")

            conn.commit()
            logging.info(f"✅ Preloaded {inserted_count} jobs successfully")

        return True

    except Exception as e:
        logging.error(f"❌ Error preloading jobs from CSV: {e}", exc_info=True)
        return False

    
async def index_cv_extracts(skills: list, aspirations: str, experience: str, education: str, cv_id: int) -> bool:
    if not isinstance(cv_id, int):
        raise ValueError("cv_id must be an integer")
    try:
        content = (
            f"Skills: {json.dumps(skills, ensure_ascii=False)} "
            f"Aspirations: {aspirations} "
            f"Experience: {experience} "
            f"Education: {education}"
        )
        doc = Document(page_content=content, metadata={"cv_id": cv_id})
        vectorstore = get_vectorstore()
        vectorstore.add_documents([doc])
        logging.info(f"Indexed CV {cv_id} into Chroma")
        return True
    except Exception as e:
        logging.error(f"Error indexing CV {cv_id}: {e}")
        return False

def delete_cv_from_chroma(cv_id: int) -> bool:
    if not isinstance(cv_id, int):
        raise ValueError("cv_id must be an integer")
    try:
        vectorstore = get_vectorstore()
        docs = vectorstore.get(where={"cv_id": cv_id})
        if not docs['ids']:
            logging.info(f"No documents found for CV {cv_id}")
            return False
        vectorstore._collection.delete(where={"cv_id": cv_id})
        logging.info(f"Deleted CV {cv_id} from Chroma")
        return True
    except Exception as e:
        logging.error(f"Error deleting CV {cv_id} from Chroma: {e}")
        return False