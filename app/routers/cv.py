# app/routers/cv.py
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import Dict, List
from models.core import DocumentInfo, DeleteFileRequest
from models.responses import CVInsightsResponse, CVImproveResponse, ImprovementSuggestion
from services.db_utils import (
    get_db_connection, insert_cv_record, get_all_cvs, delete_cv_record,
    get_cv_insights, save_cv_insights
)
from services.chroma_utils import index_cv_extracts, delete_cv_from_chroma
from services.ai_analysis import analyze_cv_insights, generate_cv_improvements
from utils.pdf_parser import extract_text_from_pdf, extract_cv_info
from utils.date_utils import normalize_date
import os
import json
import logging
from datetime import datetime

router = APIRouter()

@router.post("/upload")
async def upload_cv(file: UploadFile = File(...)):
    """Tải lên CV PDF, trích xuất thông tin, lưu vào cv_store và Chroma."""
    if not file.filename.lower().endswith('.pdf'): # type: ignore
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    max_size = 10 * 1024 * 1024  # 10MB
    if file.size > max_size: # type: ignore
        raise HTTPException(status_code=400, detail="File size exceeds 10MB")
    temp_file_path = f"temp_{file.filename}"
    file_data = None
    try:
        # Read file data
        file_data = await file.read()
        # Save to temp file for processing
        with open(temp_file_path, "wb") as buffer:
            buffer.write(file_data)
        cv_text = extract_text_from_pdf(temp_file_path)
        cv_info = extract_cv_info(cv_text)
        skills = cv_info.get("skills", [])
        aspirations = cv_info.get("career_objective", "")
        education = cv_info.get("education", [])
        experience = cv_info.get("experience", [])
        if not skills and not aspirations:
            raise HTTPException(status_code=400, detail="No skills or career objective found")
        # Tạo tóm tắt experience để index
        experience_summary = "\n".join([
            f"{exp.get('title', 'Unknown')} at {exp.get('company', 'Unknown')} ({exp.get('start_date', '')}-{exp.get('end_date', '')}): {exp.get('description', '')}"
            for exp in experience
        ]) if experience else "No experience provided"
        # Insert CV record with file_data
        cv_id = insert_cv_record(file.filename, cv_info, file_data) # type: ignore
        if not cv_id:
            raise HTTPException(status_code=500, detail="Failed to generate cv_id from database")
        try:
            await index_cv_extracts(skills, aspirations, experience_summary, education, cv_id)
        except Exception as e:
            delete_cv_record(cv_id)
            raise HTTPException(status_code=500, detail=f"Failed to index CV to Chroma: {str(e)}")
        logging.info(f"Uploaded and indexed CV {cv_id}: {file.filename}")
        return {
            "message": f"CV {file.filename} uploaded and indexed",
            "cv_id": cv_id,
            "cv_info": cv_info
        }
    except Exception as e:
        logging.error(f"Error uploading CV {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload CV: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@router.get("/list", response_model=List[DocumentInfo])
async def list_cvs(page: int = 1, page_size: int = 10):
    """Liệt kê tất cả CV trong cv_store với phân trang."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, filename, cv_info_json, upload_timestamp FROM cv_store ORDER BY upload_timestamp DESC LIMIT ? OFFSET ?",
                (page_size, (page - 1) * page_size)
            )
            cvs = [
                {
                    "id": row["id"],
                    "filename": row["filename"],
                    "cv_info_json": row["cv_info_json"],
                    "upload_timestamp": row["upload_timestamp"]
                }
                for row in cursor.fetchall()
            ]
            logging.info(f"Lấy được {len(cvs)} CV")
            return [DocumentInfo(**cv) for cv in cvs]
    except Exception as e:
        logging.error(f"Lỗi khi liệt kê CV: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Không thể liệt kê CV: {str(e)}")

@router.post("/delete")
async def delete_cv(request: DeleteFileRequest):
    """Xóa CV khỏi cv_store và Chroma."""
    if not isinstance(request.file_id, int):
        raise HTTPException(status_code=400, detail="file_id must be an integer")
    try:
        deleted = delete_cv_record(request.file_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"CV {request.file_id} không tìm thấy")
        deleted_chroma = delete_cv_from_chroma(request.file_id)
        logging.info(f"Đã xóa CV {request.file_id} khỏi cv_store và Chroma")
        return {"message": f"CV {request.file_id} đã được xóa"}
    except Exception as e:
        logging.error(f"Lỗi khi xóa CV {request.file_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Không thể xóa CV: {str(e)}")

@router.get("/{cv_id}/insights", response_model=CVInsightsResponse)
async def get_cv_insights_endpoint(cv_id: int):
    """
    Phân tích CV chuyên sâu - Đánh giá chất lượng, điểm mạnh/yếu
    Khác với /upload-cv (chỉ parse thông tin cơ bản),
    endpoint này phân tích và đánh giá CV bằng AI.
    """
    try:
        # Kiểm tra CV tồn tại
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT cv_info_json FROM cv_store WHERE id = ?", (cv_id,))
            row = cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail=f"CV {cv_id} không tìm thấy")
            cv_info = json.loads(row["cv_info_json"])
        # Kiểm tra cache
        cached_insights = get_cv_insights(cv_id)
        if cached_insights:
            logging.info(f"✅ Lấy insights từ cache cho CV {cv_id}")
            return CVInsightsResponse(
                cv_id=cv_id,
                quality_score=cached_insights['quality_score'],
                completeness={
                    "has_portfolio": False,
                    "has_certifications": False,
                    "has_projects": False,
                    "missing_sections": cached_insights['missing_sections']
                },
                market_fit={
                    "skill_match_rate": cached_insights['market_fit_score'],
                    "experience_level": "Junior",
                    "salary_range": "8-12 triệu",
                    "competitive_score": cached_insights['completeness_score'] * 10
                },
                strengths=cached_insights['strengths'],
                weaknesses=cached_insights['weaknesses'],
                last_analyzed=cached_insights['last_analyzed']
            )
        # Phân tích mới bằng AI
        logging.info(f"🔍 Bắt đầu phân tích CV {cv_id}...")
        insights = await analyze_cv_insights(cv_info)
        # Lưu vào cache
        save_cv_insights(cv_id, insights)
        logging.info(f"✅ Phân tích CV {cv_id} hoàn tất")
        return CVInsightsResponse(
            cv_id=cv_id,
            quality_score=insights.get('quality_score', 5.0),
            completeness={
                "has_portfolio": insights.get('has_portfolio', False),
                "has_certifications": insights.get('has_certifications', False),
                "has_projects": insights.get('has_projects', False),
                "missing_sections": insights.get('missing_sections', [])
            },
            market_fit={
                "skill_match_rate": insights.get('market_fit_score', 0.5),
                "experience_level": insights.get('experience_level', 'Unknown'),
                "salary_range": insights.get('salary_range', 'N/A'),
                "competitive_score": insights.get('competitive_score', 5.0)
            },
            strengths=insights.get('strengths', []),
            weaknesses=insights.get('weaknesses', []),
            last_analyzed=datetime.now().isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"❌ Lỗi phân tích CV {cv_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi phân tích CV: {str(e)}")

@router.post("/improve", response_model=CVImproveResponse)
async def improve_cv_endpoint(cv_id: int):
    """
    Gợi ý cải thiện CV cụ thể
    Dựa trên kết quả phân tích từ /cv/{cv_id}/insights,
    endpoint này đưa ra các gợi ý hành động cụ thể.
    """
    try:
        # Lấy CV info
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT cv_info_json FROM cv_store WHERE id = ?", (cv_id,))
            row = cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail=f"CV {cv_id} không tìm thấy")
            cv_info = json.loads(row["cv_info_json"])
        # Lấy insights (hoặc phân tích mới)
        insights = get_cv_insights(cv_id)
        if not insights:
            logging.info(f"Chưa có insights, phân tích CV {cv_id} trước...")
            insights_data = await analyze_cv_insights(cv_info)
            save_cv_insights(cv_id, insights_data)
            insights = insights_data
        # Tạo gợi ý cải thiện
        logging.info(f"💡 Tạo gợi ý cải thiện cho CV {cv_id}...")
        improvements = await generate_cv_improvements(cv_info, insights)
        improvement_suggestions = [
            ImprovementSuggestion(**imp) for imp in improvements
        ]
        logging.info(f"✅ Tạo {len(improvement_suggestions)} gợi ý cải thiện")
        return CVImproveResponse(
            cv_id=cv_id,
            improvements=improvement_suggestions
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Lỗi tạo gợi ý cải thiện CV {cv_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi tạo gợi ý: {str(e)}")

@router.get("/")
async def get_all_cvs_simple():
    """
    Lấy tất cả CVs với thông tin đã parse (cho frontend dashboard)
    Khác với /list-cvs (có phân trang), endpoint này trả về tất cả CVs
    với cv_info đã được parse thành object (không phải JSON string)
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, filename, cv_info_json, upload_timestamp FROM cv_store ORDER BY upload_timestamp DESC")
            rows = cursor.fetchall()
            cvs = []
            for row in rows:
                cv_info = json.loads(row["cv_info_json"]) if row["cv_info_json"] else {}
                cvs.append({
                    "id": row["id"],
                    "filename": row["filename"],
                    "cv_info": cv_info,  # Already parsed object
                    "upload_timestamp": row["upload_timestamp"]
                })
            logging.info(f"✅ Lấy {len(cvs)} CVs cho frontend")
            return cvs
    except Exception as e:
        logging.error(f"❌ Lỗi lấy CVs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi lấy CVs: {str(e)}")