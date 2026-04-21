# app/services/cv_service.py
"""
CV Analysis Service
Handles CV analysis and Q&A with uploaded CVs
"""

import logging
from typing import Optional, Dict, Any
from app.services.db_utils import get_db_connection
from app.services.llm_service import get_llm_service
from app.services.chroma_utils import get_vectorstore

logger = logging.getLogger(__name__)


async def analyze_cv_with_query(cv_id: str, query: str) -> str:
    """
    Analyze CV and answer a question about it.
    
    Args:
        cv_id: ID of the uploaded CV
        query: Question or analysis request about the CV
        
    Returns:
        Analysis result as a string
    """
    try:
        # 1. Retrieve CV info from database
        cv_data = _get_cv_from_db(cv_id)
        if not cv_data:
            return f"Không tìm thấy CV với ID: {cv_id}"
        
        # 2. Retrieve relevant context from ChromaDB using vector search
        cv_context = _retrieve_cv_context(cv_id, query)
        
        # 3. Build prompt for LLM
        prompt = _build_cv_analysis_prompt(cv_data, cv_context, query)
        
        # 4. Get LLM response with timeout handling
        logger.info(f"[CV ANALYSIS] Calling LLM for CV ID {cv_id}...")
        llm_service = get_llm_service()
        
        try:
            response = llm_service.generate_response(
                prompt,
                timeout_override=45  # CV analysis may take longer (45 seconds max)
            )
            logger.info(f"[CV ANALYSIS] ✅ Analysis completed for {cv_id}")
            return response
            
        except Exception as llm_error:
            logger.warning(f"[CV ANALYSIS] ⚠️ LLM failed ({type(llm_error).__name__}): {str(llm_error)[:150]}")
            logger.info(f"[CV ANALYSIS] Falling back to simple analysis...")
            # Fallback: Simple analysis without LLM
            try:
                fallback_response = _simple_cv_analysis_fallback(cv_data, query)
                logger.info(f"[CV ANALYSIS] ✅ Fallback analysis successful")
                return fallback_response
            except Exception as fallback_error:
                logger.error(f"[CV ANALYSIS] ❌ Fallback also failed: {str(fallback_error)}")
                return f"⚠️ CV analysis tạm thời không khả dụng. Lỗi: {str(llm_error)[:100]}"
        
    except Exception as e:
        logger.error(f"[CV ANALYSIS] ❌ Analysis failed: {str(e)}")
        raise


def _simple_cv_analysis_fallback(cv_data: Dict[str, Any], query: str) -> str:
    """
    Simple fallback analysis when LLM is unavailable
    Provides basic CV insights without requiring LLM
    """
    cv_text = cv_data.get("raw_text", "")
    filename = cv_data.get("filename", "CV")
    
    logger.info(f"[CV ANALYSIS] Using fallback analysis (no LLM)")
    
    # Simple text analysis
    response_parts = [
        f"📄 Phân tích CV: {filename}",
        "",
        "Câu hỏi: " + query,
        "",
        "📋 Thông tin từ CV:"
    ]
    
    # Extract lines with keywords
    lines = cv_text.split('\n')
    
    for keyword in ['skill', 'experience', 'education', 'name', 'email', 'phone']:
        for line in lines:
            if keyword.lower() in line.lower():
                response_parts.append(f"  • {line.strip()}")
    
    if len(response_parts) <= 5:
        response_parts.append(f"  • CV text length: {len(cv_text)} characters")
    
    response_parts.extend([
        "",
        "⚠️  Note: LLM không khả dụng. Vui lòng thử lại sau hoặc kiểm tra Ollama đang chạy."
    ])
    
    return "\n".join(response_parts)


def _get_cv_from_db(cv_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve CV text from database"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT cv_info_json, filename FROM cv_store WHERE id = ?",
                (cv_id,)
            )
            result = cursor.fetchone()
            
            if result:
                import json
                cv_data = json.loads(result[0]) if isinstance(result[0], str) else result[0]
                
                # Extract raw text - could be either new format (raw_text) or old format (extracted info)
                cv_text = cv_data.get("raw_text", "")
                if not cv_text:
                    # Fallback: if old format with extracted info, reconstruct text
                    logger.info(f"[CV] Using legacy extracted info format for {cv_id}")
                    cv_text = _reconstruct_cv_text_from_info(cv_data)
                
                return {
                    "id": cv_id,
                    "filename": result[1],
                    "raw_text": cv_text
                }
            return None
    except Exception as e:
        logger.error(f"Error fetching CV from DB: {e}")
        return None


def _reconstruct_cv_text_from_info(cv_info: Dict[str, Any]) -> str:
    """
    Reconstruct CV text from extracted information (for backward compatibility)
    """
    parts = []
    
    if cv_info.get("name"):
        parts.append(f"Name: {cv_info['name']}")
    if cv_info.get("email"):
        parts.append(f"Email: {cv_info['email']}")
    if cv_info.get("phone"):
        parts.append(f"Phone: {cv_info['phone']}")
    
    if cv_info.get("career_objective"):
        parts.append(f"\nCareer Objective: {cv_info['career_objective']}")
    
    if cv_info.get("skills"):
        parts.append(f"\nSkills: {', '.join(cv_info['skills'])}")
    
    if cv_info.get("experience"):
        parts.append("\nExperience:")
        for exp in cv_info["experience"]:
            parts.append(f"- {exp.get('title', '')} at {exp.get('company', '')} ({exp.get('start_date', '')} to {exp.get('end_date', '')})")
            if exp.get('description'):
                parts.append(f"  {exp['description']}")
    
    if cv_info.get("education"):
        parts.append("\nEducation:")
        for edu in cv_info["education"]:
            parts.append(f"- {edu.get('degree', '')} in {edu.get('major', '')} from {edu.get('school', '')}")
    
    return "\n".join(parts)


def _retrieve_cv_context(cv_id: str, query: str, top_k: int = 5) -> str:
    """
    Retrieve relevant CV content from ChromaDB based on query
    """
    try:
        vectorstore = get_vectorstore("cvs")
        
        # Search for relevant documents in CV collection
        results = vectorstore.similarity_search_with_score(query, k=top_k)
        
        # Filter results for this specific CV
        relevant_docs = []
        for doc, score in results:
            # Check if this document belongs to the CV we're analyzing
            if doc.metadata.get("cv_id") == cv_id:
                relevant_docs.append(doc.page_content)
        
        if relevant_docs:
            return "\n".join(relevant_docs)
        else:
            logger.warning(f"No relevant CV context found for {cv_id}")
            return ""
            
    except Exception as e:
        logger.error(f"Error retrieving CV context: {e}")
        return ""


def _build_cv_analysis_prompt(cv_data: Dict[str, Any], context: str, query: str) -> str:
    """Build prompt for CV analysis using raw CV text"""
    
    # Use raw CV text - let LLM do all the analysis
    cv_text = cv_data.get("raw_text", "")
    filename = cv_data.get("filename", "CV")
    
    prompt = f"""Bạn là một chuyên gia phân tích CV chuyên nghiệp. Dựa vào nội dung CV được cung cấp, hãy trả lời câu hỏi sau một cách chuyên nghiệp, chi tiết và có giá trị.

**NỘI DUNG CV ({filename}):**
{cv_text}

**NGỮ CẢNH BỔ SUNG (nếu có):**
{context if context else 'Không có'}

**CÂU HỎI VỀ CV:**
{query}

Vui lòng trả lời câu hỏi trên dựa vào thông tin CV được cung cấp. Hãy cụ thể, chi tiết, và đưa ra các gợi ý hữu ích nếu cần thiết. Nếu thông tin không có trong CV, hãy nói rõ điều đó."""
    
    return prompt


