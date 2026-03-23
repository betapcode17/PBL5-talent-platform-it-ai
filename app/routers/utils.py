# app/routers/utils.py
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response
from typing import Dict, Any, Optional
from app.models.responses import (
    QuestionSuggestion, SuggestQuestionsInput, SuggestQuestionsResponse, DocumentPreviewResponse
)
from app.services.db_utils import get_db_connection, get_document_preview, save_document_preview
from app.services.ai_analysis import generate_question_suggestions
import os
import json
import logging
from datetime import datetime

router = APIRouter()

@router.get("/preview-doc/{file_id}")
async def preview_document_pdf(file_id: int):
    """
    Serve PDF file để preview trong browser
    """
    try:
        # Lấy thông tin CV
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT filename, file_data FROM cv_store WHERE id = ?", (file_id,))
            row = cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail=f"File {file_id} không tìm thấy")
        # Nếu không có file_data (CV cũ), trả về placeholder
        if not row['file_data']:
            logging.warning(f" CV {file_id} không có file_data. Trả về placeholder.")
            return Response(
                content=f"<html><body><h3>CV Preview không khả dụng</h3><p>File: {row['filename']}</p><p>Vui lòng upload lại CV để xem preview.</p></body></html>",
                media_type="text/html"
            )
        # Tạo thư mục temp nếu chưa có
        temp_dir = os.path.join(os.path.dirname(__file__), "..", "..", "temp_pdfs")  # Adjust path for structure
        os.makedirs(temp_dir, exist_ok=True)
        # Lưu PDF vào temp file
        temp_file_path = os.path.join(temp_dir, f"cv_{file_id}.pdf")
        with open(temp_file_path, 'wb') as f:
            f.write(row['file_data'])
        logging.info(f" Tạo preview PDF cho file {file_id}")
        # Trả về PDF file với header inline để hiển thị trong browser
        return FileResponse(
            temp_file_path,
            media_type='application/pdf',
            headers={
                "Content-Disposition": f"inline; filename={row['filename']}"
            }
        )
    except Exception as e:
        logging.error(f" Lỗi preview PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi preview: {str(e)}")

@router.get("/preview-doc-info/{file_id}", response_model=DocumentPreviewResponse)
async def preview_document_info(file_id: int):
    """
    Xem trước tài liệu (CV) - Hiển thị summary, thumbnail
    Endpoint mới để preview CV trước khi submit hoặc xem lại.
    """
    try:
        # Kiểm tra cache
        cached_preview = get_document_preview(file_id)
        if cached_preview:
            logging.info(f" Lấy preview từ cache cho file {file_id}")
            # Lấy thông tin CV
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT filename, cv_info_json FROM cv_store WHERE id = ?", (file_id,))
                row = cursor.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail=f"File {file_id} không tìm thấy")
                cv_info = json.loads(row["cv_info_json"])
                return DocumentPreviewResponse(
                    file_id=file_id,
                    type=cached_preview['type'],
                    filename=row['filename'],
                    preview={
                        "title": f"CV - {cv_info.get('name', 'Unknown')}",
                        "summary": cached_preview['summary'],
                        "page_count": cached_preview['page_count'],
                        "file_size": f"{cached_preview['file_size'] / 1024:.1f} KB" if cached_preview['file_size'] else "N/A"
                    },
                    quick_info={
                        "name": cv_info.get('name', 'N/A'),
                        "email": cv_info.get('email', 'N/A'),
                        "phone": cv_info.get('phone', 'N/A'),
                        "top_skills": cv_info.get('skills', [])[:5]
                    }
                )
        # Tạo preview mới
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT filename, cv_info_json FROM cv_store WHERE id = ?", (file_id,))
            row = cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail=f"File {file_id} không tìm thấy")
            cv_info = json.loads(row["cv_info_json"])
            # Tạo summary
            skills_str = ", ".join(cv_info.get('skills', [])[:5])
            summary = f"{cv_info.get('name', 'Unknown')} - {skills_str}"
            # Lưu preview
            preview_data = {
                "type": "cv",
                "summary": summary,
                "page_count": 1,  # Placeholder
                "file_size": 0  # Placeholder
            }
            save_document_preview(file_id, preview_data)
            logging.info(f" Tạo preview mới cho file {file_id}")
            return DocumentPreviewResponse(
                file_id=file_id,
                type="cv",
                filename=row['filename'],
                preview={
                    "title": f"CV - {cv_info.get('name', 'Unknown')}",
                    "summary": summary,
                    "page_count": 1,
                    "file_size": "N/A"
                },
                quick_info={
                    "name": cv_info.get('name', 'N/A'),
                    "email": cv_info.get('email', 'N/A'),
                    "phone": cv_info.get('phone', 'N/A'),
                    "top_skills": cv_info.get('skills', [])[:5]
                }
            )
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f" Lỗi preview document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi preview: {str(e)}")

@router.post("/suggest-questions", response_model=SuggestQuestionsResponse)
async def suggest_questions_endpoint(input: SuggestQuestionsInput):
    """
    Gợi ý câu hỏi dựa trên context
    Endpoint mới để hướng dẫn user hỏi AI những câu hỏi phù hợp.
    """
    try:
        context = input.context
        cv_id = input.cv_id
        job_id = input.job_id
        # Lấy CV info nếu có
        cv_info = None
        if cv_id:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT cv_info_json FROM cv_store WHERE id = ?", (cv_id,))
                row = cursor.fetchone()
                if row:
                    cv_info = json.loads(row["cv_info_json"])
        # Lấy job info nếu có
        job_info = None
        if job_id:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT job_title FROM job_store WHERE id = ?", (job_id,))
                row = cursor.fetchone()
                if row:
                    job_info = {"job_title": row["job_title"]}
        # Tạo gợi ý câu hỏi
        suggestions = generate_question_suggestions(context, cv_info, job_info)
        question_items = [
            QuestionSuggestion(**q) for q in suggestions
        ]
        logging.info(f" Tạo {len(question_items)} gợi ý câu hỏi cho context '{context}'")
        return SuggestQuestionsResponse(suggestions=question_items)
    except Exception as e:
        logging.error(f" Lỗi tạo gợi ý câu hỏi: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi tạo gợi ý: {str(e)}")