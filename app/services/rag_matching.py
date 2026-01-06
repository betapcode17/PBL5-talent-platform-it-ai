# app/services/rag_matching.py
"""
Core RAG matching logic for CV-Job matching.
Runnable-based (NO deprecated chains).
"""

import json
import logging
from typing import List, Tuple, Dict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from services.api_key_manager import get_next_api_key
from services.chroma_utils import get_vectorstore
from services.rag_helpers import _to_int_job_id, _prefix_doc_with_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -------------------------------------------------------------------
# RAG COMPONENTS
# -------------------------------------------------------------------

def get_rag_components(model: str = "gemini-2.5-flash"):
    """
    Trả về:
      - retriever
      - rewrite_chain (rewrite query từ CV + history)
      - qa_chain (Runnable: prompt -> LLM -> JSON)
    """

    google_api_key = get_next_api_key()
    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=google_api_key,
        temperature=0.2,
    )

    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

    # ---------- Rewrite query (history-aware thay thế) ----------
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an assistant helping to match CV skills, experience, education and career goals "
            "with job postings. Reformulate the CV information into a concise job-search query. "
            "Do NOT answer, only output the rewritten query."
        ),
        MessagesPlaceholder("match_history"),
        ("human", "CV input: {input}")
    ])

    rewrite_chain = contextualize_q_prompt | llm | RunnableLambda(lambda x: x.content)

    # ---------- QA Prompt ----------
    qa_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a job matching assistant.\n"
            "Use ONLY the provided job postings context.\n"
            "Select top 5 most relevant jobs.\n"
            "Weights: 50% skills, 30% experience, 10% aspirations, 10% education.\n\n"
            "IMPORTANT RULES:\n"
            "- job_id MUST come from 'JOB_ID:' in context\n"
            "- If job_id is not numeric → skip\n"
            "- Do NOT invent jobs\n\n"
            "Return STRICT JSON:\n"
            "{\n"
            ' "matched_jobs": [{\n'
            '   "job_id": int,\n'
            '   "job_title": str,\n'
            '   "job_url": str,\n'
            '   "match_score": float,\n'
            '   "matched_skills": [str],\n'
            '   "matched_aspirations": [str],\n'
            '   "matched_experience": [str],\n'
            '   "matched_education": [str],\n'
            '   "why_match": str\n'
            " }],\n"
            ' "suggestions": [{"skill_or_experience": str, "suggestion": str}]\n'
            "}"
        ),
        ("system", "Context (job postings):\n{context}"),
        MessagesPlaceholder("match_history"),
        ("human", "CV input: {input}")
    ])

    qa_chain = qa_prompt | llm | JsonOutputParser()

    logging.info("✅ RAG components (Runnable-based) ready")
    return retriever, rewrite_chain, qa_chain


# -------------------------------------------------------------------
# MATCH CV
# -------------------------------------------------------------------

async def match_cv(cv: dict, filtered_job_ids: List[int], session_id: str) -> dict:
    try:
        cv_id = cv.get("cv_id")
        if not cv_id:
            raise ValueError("CV must include cv_id")

        query = (
            f"Skills: {json.dumps(cv.get('skills', []), ensure_ascii=False)} "
            f"Aspirations: {cv.get('aspirations', '')} "
            f"Experience: {cv.get('experience', '')} "
            f"Education: {cv.get('education', '')}"
        )

        retriever, rewrite_chain, qa_chain = get_rag_components()
        vectorstore = get_vectorstore()

        # ---------- Rewrite query ----------
        rewritten_query = await rewrite_chain.ainvoke({
            "input": query,
            "match_history": [],
        })

        logging.info(f"🧠 Rewritten query:\n{rewritten_query}")

        # ---------- Collect context docs ----------
        docs: List[Document] = []

        if filtered_job_ids:
            raw = vectorstore.get(where={"job_id": {"$in": [str(i) for i in filtered_job_ids]}})
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
            for d in retriever.get_relevant_documents(rewritten_query):
                docs.append(_prefix_doc_with_id(d))

        logging.info(f"✅ Retrieved {len(docs)} job documents")

        # ---------- QA ----------
        result = await qa_chain.ainvoke({
            "context": docs,
            "input": query,
            "match_history": [],
        })

        output = result if isinstance(result, dict) else {}
        output["cv_id"] = cv_id

        normalized_jobs = []
        for job in output.get("matched_jobs", []) or []:
            job_id = _to_int_job_id(job.get("job_id"))
            if job_id is None:
                continue

            normalized_jobs.append({
                "job_id": job_id,
                "job_title": job.get("job_title", ""),
                "job_url": job.get("job_url", ""),
                "match_score": float(job.get("match_score", 0)),
                "matched_skills": job.get("matched_skills", []),
                "matched_aspirations": job.get("matched_aspirations", []),
                "matched_experience": job.get("matched_experience", []),
                "matched_education": job.get("matched_education", []),
            })

        output["matched_jobs"] = normalized_jobs

        if not normalized_jobs:
            output.setdefault("suggestions", []).append({
                "skill_or_experience": "N/A",
                "suggestion": "No valid job matches found"
            })

        logging.info(f"🎯 CV {cv_id} matched {len(normalized_jobs)} jobs")
        return output

    except Exception as e:
        logging.error(f"❌ CV match failed: {e}")
        return {
            "cv_id": cv.get("cv_id", "unknown"),
            "matched_jobs": [],
            "suggestions": [{
                "skill_or_experience": "N/A",
                "suggestion": str(e)
            }]
        }
