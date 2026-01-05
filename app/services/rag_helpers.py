# app/services/rag_helpers.py
"""
Helper functions for RAG matching.
Contains: _to_int_job_id, _prefix_doc_with_id, verify_job_id_consistency
"""

import re
import json
import logging
from langchain_core.documents import Document
from typing import List
from app.services.chroma_utils import get_vectorstore
from app.services.db_utils import get_db_connection

def _to_int_job_id(x):
    """Chuyển job_id về int an toàn (nhận int, '716', 'job_716'...)."""
    if isinstance(x, int):
        return x
    if isinstance(x, str):
        m = re.search(r"\d+", x)
        if m:
            try:
                return int(m.group())
            except Exception:
                return None
    return None

def _prefix_doc_with_id(doc: Document) -> Document:
    """Nhét JOB_ID/TITLE/URL vào đầu page_content và RÚT GỌN content để Gemini xử lý nhanh hơn."""
    mid = doc.metadata or {}
    job_id = mid.get("job_id", "")
    job_title = mid.get("job_title", "")
    job_url = mid.get("job_url", "")
    # Rút gọn content: chỉ lấy 800 ký tự đầu (đủ cho matching)
    content = doc.page_content or ""
    if len(content) > 800:
        content = content[:800] + "..."
    header = f"JOB_ID: {job_id}\nJOB_TITLE: {job_title}\nJOB_URL: {job_url}\n-----\n"
    doc.page_content = header + content
    return doc

def verify_job_id_consistency(job_id: int) -> bool:
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT job_url, job_title, work_location, skills
                FROM job_store
                WHERE id = ?
            ''', (job_id,))
            sqlite_job = cursor.fetchone()
            if not sqlite_job:
                logging.error(f"No job found in SQLite for job_id {job_id}")
                return False
            sqlite_data = dict(sqlite_job)
            sqlite_data["job_id"] = job_id
            logging.info(f"SQLite data for job_id {job_id}: {sqlite_data}")
        vectorstore = get_vectorstore()
        chroma_docs = vectorstore.get(where={"job_id": str(job_id)})
        if not chroma_docs['ids']:
            logging.error(f"No document found in Chroma for job_id {job_id}")
            return False
        chroma_metadata = chroma_docs['metadatas'][0]
        chroma_data = {
            "job_id": chroma_metadata.get("job_id", None),
            "job_url": chroma_metadata.get("job_url", ""),
            "job_title": chroma_metadata.get("job_title", ""),
            "work_location": chroma_metadata.get("work_location", ""),
            "skills": chroma_metadata.get("skills", "")
        }
        logging.info(f"Chroma data for job_id {job_id}: {chroma_data}")
        fields_to_compare = ["job_url", "job_title", "work_location", "skills"]
        is_consistent = True
        for field in fields_to_compare:
            sqlite_value = sqlite_data[field]
            chroma_value = chroma_data[field]
            if field == "skills" and sqlite_value:
                try:
                    sqlite_value = json.dumps(json.loads(sqlite_value), ensure_ascii=False)
                except json.JSONDecodeError:
                    pass
            if sqlite_value != chroma_value:
                logging.error(f"Mismatch in {field} for job_id {job_id}: SQLite={sqlite_value}, Chroma={chroma_value}")
                is_consistent = False
        if is_consistent:
            logging.info(f"✅ job_id {job_id} is consistent between SQLite and Chroma")
        else:
            logging.warning(f"⚠️ job_id {job_id} is NOT consistent between SQLite and Chroma")
        return is_consistent
    except Exception as e:
        logging.error(f"Error verifying job_id {job_id}: {e}")
        return False