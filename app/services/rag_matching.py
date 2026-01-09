

import json
import logging
from typing import List, Dict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document

from services.api_key_manager import get_next_api_key
from services.chroma_utils import get_vectorstore
from services.rag_helpers import _to_int_job_id, _prefix_doc_with_id
from prompts import rewrite_prompt, qa_prompt 

logging.basicConfig(level=logging.INFO)


def get_rag_components(model="gemini-2.5-flash"):
    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=get_next_api_key(),
        temperature=0.2,
    )

    retriever = get_vectorstore().as_retriever(search_kwargs={"k": 10})

    # ---------- Rewrite chain ----------
    rewrite_chain = rewrite_prompt | llm | RunnableLambda(lambda x: x.content)

    # ---------- QA chain ----------
    qa_chain = qa_prompt | llm | JsonOutputParser()
    return retriever, rewrite_chain, qa_chain


async def match_cv(cv: dict, filtered_job_ids: List[int], session_id: str):
    try:
        query = json.dumps(cv, ensure_ascii=False)

        retriever, rewrite_chain, qa_chain = get_rag_components()
        vectorstore = get_vectorstore()

        rewritten = await rewrite_chain.ainvoke({"input": query})

        if filtered_job_ids:
            raw = vectorstore.get(where={"job_id": {"$in": [str(i) for i in filtered_job_ids]}})
            docs = [
                _prefix_doc_with_id(
                    Document(page_content=m.get("content", ""), metadata=m)
                )
                for m in raw.get("metadatas", [])
                if str(m.get("job_id", "")).isdigit()
            ]
        else:
            docs = retriever.invoke(rewritten)
            docs = [_prefix_doc_with_id(d) for d in docs]

        result = await qa_chain.ainvoke({
            "context": docs,
            "input": query
        })

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