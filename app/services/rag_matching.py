

import json
import logging
from typing import List, Dict

from app.services.rag_helpers import _to_int_job_id, _prefix_doc_with_id

logging.basicConfig(level=logging.INFO)



async def match_cv(cv: dict, filtered_job_ids: List[int], session_id: str):
    """Match CV to jobs using Ollama LLM and ChromaDB retrieval."""
    try:
        from langchain_core.documents import Document  # type: ignore
        from app.services.chroma_utils import get_vectorstore
        from app.services.llm_service import get_llm_service

        # Validate input
        if not isinstance(filtered_job_ids, list):
            logging.error(f" filtered_job_ids phải là list, nhận được: {type(filtered_job_ids)}")
            filtered_job_ids = []
        
        # Ensure all elements are integers
        filtered_job_ids = [int(jid) for jid in filtered_job_ids if isinstance(jid, (int, str)) and str(jid).isdigit()]
        logging.info(f" Matching với {len(filtered_job_ids)} filtered jobs")
        
        query = json.dumps(cv, ensure_ascii=False)

        # Get LLM service (Ollama)
        llm_service = get_llm_service()
        vectorstore = get_vectorstore()
        
        # Rewrite query
        rewrite_prompt_text = f"""Rewrite this CV into a concise job search query:
{query}

Return only the rewritten query, no explanation."""
        
        rewritten = llm_service.generate_response(rewrite_prompt_text)
        logging.info(f" Rewritten query: {rewritten[:100]}...")

        # Get retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

        # Retrieve documents
        if filtered_job_ids:
            logging.info(f" Querying ChromaDB with {len(filtered_job_ids)} filtered job IDs")
            raw = vectorstore.get(where={"job_id": {"$in": [str(i) for i in filtered_job_ids]}})
            logging.info(f" ChromaDB returned {len(raw.get('metadatas', []))} documents")
            docs = [
                _prefix_doc_with_id(
                    Document(page_content=m.get("content", ""), metadata=m)
                )
                for m in raw.get("metadatas", [])
                if str(m.get("job_id", "")).isdigit()
            ]
        else:
            logging.info(f" Using retriever for semantic search (no filters)")
            docs = retriever.invoke(rewritten)
            logging.info(f" Retriever returned {len(docs)} documents")
            docs = [_prefix_doc_with_id(d) for d in docs]

        logging.info(f" Sending {len(docs)} docs to LLM for matching")
        if not docs:
            logging.error(" Không có documents để match! ChromaDB có thể chưa được index.")
            return {
                "matched_jobs": [],
                "suggestions": [{
                    "skill_or_experience": "N/A",
                    "suggestion": "Không tìm thấy công việc trong hệ thống. Vui lòng kiểm tra database và ChromaDB index."
                }]
            }
        
        # Format context for LLM
        context_text = "\n".join([f"Job {i}: {doc.page_content}" for i, doc in enumerate(docs)])
        
        qa_prompt_text = f"""You are a CV-Job matching assistant. Given a CV and job postings, select the top 5 matching jobs.

CV Information:
{query}

Job Postings:
{context_text}

Return STRICT JSON with no other text:
{{
  "matched_jobs": [{{
    "job_id": int,
    "job_title": str,
    "match_score": float (0-1),
    "explanation": str
  }}],
  "suggestions": [{{
    "skill_or_experience": str,
    "suggestion": str
  }}]
}}"""
        
        result_text = llm_service.generate_response(qa_prompt_text)
        logging.info(f" LLM response: {result_text[:200]}...")
        
        # Parse JSON
        try:
            result = json.loads(result_text)
        except json.JSONDecodeError:
            logging.warning(f" Failed to parse LLM response as JSON: {result_text[:100]}")
            result = {"matched_jobs": [], "suggestions": []}

        normalized = []
        for job in result.get("matched_jobs", []):
            jid = _to_int_job_id(job.get("job_id"))
            if jid is None:
                continue

            normalized.append({
                "job_id": jid,
                "job_title": job.get("job_title", ""),
                "job_url": job.get("job_url", ""),
                "match_score": float(job.get("match_score", 0)),
                "explanation": job.get("explanation", {})
            })

        return {
            "matched_jobs": normalized,
            "suggestions": result.get("suggestions", [])
        }

    except Exception as e:
        logging.exception("match_cv failed")
        return {
            "matched_jobs": [],
            "suggestions": [{
                "skill_or_experience": "N/A",
                "suggestion": str(e)
            }]
        }