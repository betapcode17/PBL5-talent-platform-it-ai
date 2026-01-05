# app/services/rag_matching.py
"""
Core RAG matching logic for CV-Job matching.
Contains: get_rag_components, match_cv
"""

import json
import logging
from typing import List, Tuple, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
import asyncio

from app.services.api_key_manager import get_next_api_key
from app.services.chroma_utils import get_vectorstore
from app.services.rag_helpers import _to_int_job_id, _prefix_doc_with_id

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_rag_components(model: str = "gemini-2.5-flash") -> Tuple:
    """
    Trả về (retriever, qa_chain, qa_prompt).
    Sau đó ta sẽ tự gọi retriever -> lấy docs -> gọi thẳng qa_chain với {context: docs}.
    """
    # Use API key rotation
    google_api_key = get_next_api_key()
    llm = ChatGoogleGenerativeAI(model=model, google_api_key=google_api_key)
    # Tạo retriever từ Chroma - lấy 20 jobs để Gemini rank
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
    # Prompt tạo query tóm tắt CV thành truy vấn tìm việc (cho history-aware retriever nếu dùng)
    contextualize_q_system_prompt = (
        "You are an assistant helping to match CV skills, aspirations, experience, and education with job postings.\n"
        "Given the match history and a combined input of skills, aspirations, experience, and education from a CV, "
        "reformulate them into a concise query for job matching.\n"
        "Extract and prioritize key keywords from skills, experience, aspirations, and education.\n"
        "Ensure the query focuses on matching CV skills and experience with job description, candidate requirements, and skills listed in job postings.\n"
        "Do NOT generate an answer, only reformulate the input into a clear and concise query."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("match_history"),
        ("human", "Match jobs for CV with input: {input}")
    ])
    # Bạn có thể dùng history_aware_retriever nếu cần, hiện tại ta không dùng nó để giữ chủ động context
    _ = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    # Prompt chính để RAG đánh giá và match
    qa_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a job matching assistant. Your task is to match CV skills, aspirations, experience, and education with job postings.\n"
        "Use the provided context (job postings) to identify the top 5 most relevant jobs.\n"
        "Assign weights: 50% for skills, 30% for experience, 10% for aspirations, 10% for education.\n"
        "For matched_skills, ONLY include skills explicitly mentioned in the job's candidate_requirements, job_description, or skills list. "
        "Do NOT include skills from the CV that are not explicitly required or mentioned in the job context.\n"
        "Match experience by comparing CV experience with job description and experience required. "
        "Match aspirations with job title or description. Match education with education required.\n"
        "Provide suggestions to improve skills or gain experience relevant to the matched jobs, focusing on skills present in the CV but not matched.\n"
        "Return a JSON object with the following structure:\n"
        "{{\n"
        " \"matched_jobs\": [{{\n"
        " \"job_id\": int,\n"
        " \"job_title\": str,\n"
        " \"job_url\": str,\n"
        " \"match_score\": float,\n"
        " \"matched_skills\": [str],\n"
        " \"matched_aspirations\": [str],\n"
        " \"matched_experience\": [str],\n"
        " \"matched_education\": [str],\n"
        " \"why_match\": str (explain in Vietnamese why this job matches the CV, focusing on matched skills, experience, and career goals. Be specific and concise, max 100 words)\n"
        " }}],\n"
        " \"suggestions\": [{{\"skill_or_experience\": str, \"suggestion\": str}}]\n"
        "}}\n"
        "Ensure the response is concise, accurate, and based only on the provided context.\n"
        "Do not include any placeholder or sample data, mock examples — only use the actual context data.\n"
        "IMPORTANT:\n"
        "- The 'job_id' MUST be taken from the 'JOB_ID:' line in the context text (not invented).\n"
        "- If a job in the context has a 'JOB_ID' that is not a number, skip it.\n"
        "- Only pick jobs that appear in the provided context.\n"
    ),
    ("system", "Context (job postings):\n{context}"),
    MessagesPlaceholder("match_history"),
    ("human", "Match jobs for CV with input: {input}")
])
    qa_chain = create_stuff_documents_chain(llm, qa_prompt, output_parser=JsonOutputParser())
    logging.info("✅ StuffDocumentsChain created for Gemini.")
    return retriever, qa_chain, qa_prompt

async def match_cv(cv: dict, filtered_job_ids: List[int], session_id: str) -> dict:
    """
    Match một CV với danh sách job sử dụng:
      1) retriever để lấy top docs
      2) ép JOB_ID/TITLE/URL vào page_content của từng doc
      3) gọi thẳng qa_chain với {context: docs}
    => Tránh việc create_retrieval_chain bỏ qua context thủ công.
    """
    try:
        cv_id = cv.get("cv_id")
        if not cv_id:
            raise ValueError("CV must include cv_id")
        # ===== 1) Chuẩn bị query =====
        query = (
            f"Skills: {json.dumps(cv.get('skills', []), ensure_ascii=False)} "
            f"Aspirations: {cv.get('aspirations', '')} "
            f"Experience: {cv.get('experience', '')} "
            f"Education: {cv.get('education', '')}"
        )
        logging.info(f"\n🧠 [CV {cv_id}] Query sinh ra từ CV:\n{query}\n")
        # ===== 2) Lấy retriever & QA chain =====
        retriever, qa_chain, qa_prompt = get_rag_components()
        vectorstore = get_vectorstore()
        # ===== 3) Chuẩn bị context docs =====
        logging.info(f"🔎 Đang truy vấn retriever cho CV {cv_id} ...")
        docs: List[Document] = []
        if filtered_job_ids: # nếu có filter trước
            job_id_strs = [str(j) for j in filtered_job_ids]
            raw = vectorstore.get(where={"job_id": {"$in": job_id_strs}})
            if not raw["ids"]:
                return {
                    "cv_id": cv_id,
                    "matched_jobs": [],
                    "suggestions": [{"skill_or_experience": "N/A", "suggestion": "No jobs matched the filters"}]
                }
            for meta in raw.get("metadatas", []):
                if str(meta.get("job_id", "")).isdigit():
                    d = Document(
                        page_content=meta.get("content", ""),
                        metadata={
                            "job_id": meta.get("job_id"),
                            "job_title": meta.get("job_title"),
                            "job_url": meta.get("job_url"),
                        },
                    )
                    docs.append(_prefix_doc_with_id(d))
        else:
            # Lấy context từ retriever (nhanh & gọn)
            context_docs = retriever.get_relevant_documents(query)
            for d in context_docs:
                docs.append(_prefix_doc_with_id(d))
        # Log số lượng jobs tìm được (rút gọn logging)
        logging.info(f"✅ Tìm được {len(docs)} jobs phù hợp để gửi vào Gemini")
        # ===== 6) Gọi thẳng QA chain với context thủ công =====
        result = await qa_chain.ainvoke({
            "context": docs,
            "input": query,
            "match_history": [],
        })
        # ===== 7) Parse & normalize output =====
        output = result or {}
        if not isinstance(output, dict):
            output = {}
        output["cv_id"] = cv_id
        matched = output.get("matched_jobs", []) or []
        normalized_jobs = []
        for job in matched:
            try:
                # chuẩn hóa job_id về int
                job_id = _to_int_job_id(job.get("job_id"))
                if job_id is None:
                    logging.warning(f"⚠️ Invalid job_id trong output: {job}")
                    continue
                # ép kiểu cẩn thận
                job_title = job.get("job_title") or ""
                job_url = job.get("job_url") or ""
                match_score = float(job.get("match_score", 0.0))
                matched_skills = job.get("matched_skills") or []
                matched_asp = job.get("matched_aspirations") or []
                matched_exp = job.get("matched_experience") or []
                matched_edu = job.get("matched_education") or []
                normalized_jobs.append({
                    "job_id": job_id,
                    "job_title": job_title,
                    "job_url": job_url,
                    "match_score": match_score,
                    "matched_skills": matched_skills if isinstance(matched_skills, list) else [],
                    "matched_aspirations": matched_asp if isinstance(matched_asp, list) else [],
                    "matched_experience": matched_exp if isinstance(matched_exp, list) else [],
                    "matched_education": matched_edu if isinstance(matched_edu, list) else [],
                })
            except Exception as e:
                logging.warning(f"⚠️ Lỗi khi chuẩn hóa job: {e} | raw={job}")
        output["matched_jobs"] = normalized_jobs
        if not normalized_jobs:
            output.setdefault("suggestions", [])
            output["suggestions"].append({
                "skill_or_experience": "N/A",
                "suggestion": "No valid job_id returned by RAG"
            })
        logging.info(f"✅ CV {cv_id} matched {len(normalized_jobs)} jobs successfully")
        return output
    except Exception as e:
        logging.error(f"❌ Error matching CV {cv.get('cv_id', 'unknown')}: {e}")
        return {
            "cv_id": cv.get("cv_id", "unknown"),
            "matched_jobs": [],
            "suggestions": [{"skill_or_experience": "N/A", "suggestion": f"Failed to process CV: {e}"}],
        }