# app/services/cv_qa_simple.py

import logging
import requests
from typing import List
from PyPDF2 import PdfReader
import os

logger = logging.getLogger(__name__)


# =========================
# PDF READER
# =========================
def read_pdf(file_path: str) -> str:
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
        logger.info(f"[PDF] Extracted {len(text)} chars")
        return text
    except Exception as e:
        logger.error(f"[PDF ERROR] {e}")
        raise


# =========================
# TEXT SPLIT (CHUNKING)
# =========================
def split_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


# =========================
# SIMPLE RELEVANCE SCORING
# =========================
def select_relevant_chunks(chunks: List[str], question: str, top_k: int = 3) -> List[str]:
    question_words = question.lower().split()
    scored = []

    for chunk in chunks:
        score = sum(1 for word in question_words if word in chunk.lower())
        scored.append((score, chunk))

    scored.sort(reverse=True, key=lambda x: x[0])

    top_chunks = [chunk for score, chunk in scored[:top_k] if score > 0]

    # fallback nếu không match keyword
    if not top_chunks:
        return chunks[:top_k]

    return top_chunks


# =========================
# MAIN FUNCTION
# =========================
async def answer_cv_question(cv_id: str, question: str) -> str:
    try:
        file_path = str(cv_id)

        logger.info("========== CV ANALYSIS START ==========")
        logger.info(f"File: {file_path}")
        logger.info(f"Question: {question}")

        # Nếu chỉ là ID → tìm file
        if file_path.isdigit():
            for name in [f"cv_{cv_id}.pdf", f"{cv_id}.pdf"]:
                if os.path.exists(name):
                    file_path = name
                    break

        if not os.path.exists(file_path):
            return f"Không tìm thấy file: {file_path}"

        # =========================
        # 1. READ PDF
        # =========================
        cv_text = read_pdf(file_path)

        if not cv_text.strip():
            return "CV trống hoặc không đọc được"

        # =========================
        # 2. LIMIT SIZE (ANTI TIMEOUT)
        # =========================
        MAX_CHARS = 6000
        if len(cv_text) > MAX_CHARS:
            logger.info("[CV] Text too long → truncating")
            cv_text = cv_text[:MAX_CHARS]

        # =========================
        # 3. CHUNKING
        # =========================
        chunks = split_text(cv_text)

        # =========================
        # 4. SELECT RELEVANT
        # =========================
        relevant_chunks = select_relevant_chunks(chunks, question)

        context = "\n\n---\n\n".join(relevant_chunks)

        logger.info(f"[CV] Using {len(relevant_chunks)} chunks")

        # =========================
        # 5. BUILD PROMPT
        # =========================
        prompt = f"""
You are a professional HR.

CV CONTENT:
{context}

QUESTION:
{question}

Answer concisely and professionally.
"""

        # =========================
        # 6. CALL OLLAMA
        # =========================
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5:3b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 300,
                    "temperature": 0.7
                }
            },
            timeout=120
        )

        if response.status_code == 200:
            result = response.json()
            answer = result.get("response", "").strip()
            logger.info(f"[CV QA] Done ({len(answer)} chars)")
            return answer

        return f"Lỗi Ollama: HTTP {response.status_code}"

    except requests.exceptions.Timeout:
        return "Timeout: CV quá dài hoặc model xử lý chậm."

    except requests.exceptions.ConnectionError:
        return "Không kết nối được Ollama. Hãy chạy: ollama serve"

    except Exception as e:
        logger.error(f"[ERROR] {e}")
        return f"Lỗi xử lý CV: {str(e)}"