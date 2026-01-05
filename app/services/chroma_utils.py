import json
import logging
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from db_utils import get_db_connection, create_tables

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
            # Lấy API key từ environment
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if not google_api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")

            base_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(base_dir)
            chroma_path = os.path.join(project_root, "db/chroma_db")
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

def preload_jobs(jsonl_path: str, batch_size: int = 1000) -> bool:
    try:
        if not os.path.exists(jsonl_path):
            logging.error(f"File {jsonl_path} does not exist")
            return False
        if not jsonl_path.lower().endswith('.jsonl'):
            raise ValueError(f"File must be .jsonl, got: {jsonl_path}")

        # Tạo bảng trước khi truy vấn
        create_tables()
        logging.info("Ensured database tables are created")

        vectorstore = get_vectorstore()
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM job_store')
            if cursor.fetchone()[0] > 0:
                logging.info("Skipping preload: job_store already populated")
                return True

            documents = []
            with open(jsonl_path, 'r', encoding='utf-8-sig') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            continue
                        # Lưu vào SQLite
                        number_of_hires = int(float(str(row.get('number_of_hires', '1')).split()[0])) if row.get('number_of_hires') else 1
                        values = (
                            row.get('name', '') or '',
                            row.get('job_title', '') or '',
                            row.get('job_url', '') or '',
                            row.get('job_description', '') or '',
                            row.get('candidate_requirements', '') or '',
                            row.get('benefits', '') or '',
                            row.get('work_location', '') or '',
                            row.get('work_time', '') or '',
                            row.get('job_tags', '') or '',
                            row.get('skills', '') or '',
                            row.get('related_categories', '') or '',
                            row.get('salary', '') or '',
                            row.get('experience', '') or '',
                            row.get('deadline', '') or '',
                            row.get('company_logo', '') or '',
                            row.get('company_scale', '') or '',
                            row.get('company_field', '') or '',
                            row.get('company_address', '') or '',
                            row.get('level', '') or '',
                            row.get('education', '') or '',
                            number_of_hires,
                            row.get('work_type', '') or '',
                            row.get('company_url', '') or '',
                            row.get('timestamp', '') or ''
                        )
                        cursor.execute('''INSERT OR IGNORE INTO job_store (
                            name, job_title, job_url, job_description, candidate_requirements, 
                            benefits, work_location, work_time, job_tags, skills, 
                            related_categories, salary, experience, deadline, company_logo, 
                            company_scale, company_field, company_address, level, education, 
                            number_of_hires, work_type, company_url, timestamp
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', values)
                        if cursor.rowcount > 0:
                            cursor.execute('SELECT id FROM job_store WHERE job_url = ?', (row.get('job_url', ''),))
                            job_id = cursor.fetchone()[0]
                            row['job_id'] = job_id
                            # Tạo document đầy đủ (không cắt)
                            page_content = (
                                f"{row.get('job_title', '')} "
                                f"{row.get('job_description', '')} "
                                f"{row.get('candidate_requirements', '')} "
                                f"{json.dumps(row.get('skills', []), ensure_ascii=False)}"
                            )
                            documents.append(Document(page_content=page_content, metadata=row))
                            # Lưu vào Chroma theo batch 1000
                            if len(documents) >= batch_size:
                                vectorstore.add_documents(documents)
                                logging.info(f"Added batch of {len(documents)} jobs to Chroma")
                                documents = []
                    except json.JSONDecodeError as e:
                        logging.warning(f"Skipping line: JSON error - {e}")
                        continue
            # Lưu batch cuối nếu còn
            if documents:
                vectorstore.add_documents(documents)
                logging.info(f"Added final batch of {len(documents)} jobs to Chroma")
            conn.commit()
            logging.info(f"Preloaded jobs from {jsonl_path}")
            return True
    except Exception as e:
        logging.error(f"Error preloading jobs: {e}")
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