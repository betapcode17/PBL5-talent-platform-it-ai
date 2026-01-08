import json
import logging
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from services.api_key_manager import get_next_api_key
from services.chroma_utils import get_vectorstore
from services.rag_helpers import _to_int_job_id, _prefix_doc_with_id

logging.basicConfig(level=logging.INFO)


def get_rag_components(model="gemini-2.5-flash"):
    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=get_next_api_key(),
        temperature=0.2,
    )

    retriever = get_vectorstore().as_retriever(search_kwargs={"k": 10})

    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite CV info into a concise job search query."),
        ("human", "{input}")
    ])

    rewrite_chain = rewrite_prompt | llm | RunnableLambda(lambda x: x.content)

    qa_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a job matching assistant.
Use ONLY the provided job postings context.
Select top 5 most relevant jobs.

Weights:
- Skills: 50%
- Experience: 30%
- Aspirations: 10%
- Education: 10%

IMPORTANT:
- job_id MUST come from JOB_ID in context
- Do NOT invent data

Return STRICT JSON ONLY:
{{
  "matched_jobs": [
    {{
      "job_id": int,
      "job_title": str,
      "job_url": str,
      "match_score": float,
      "matched_skills": [str],
      "matched_aspirations": [str],
      "matched_experience": [str],
      "matched_education": [str],
      "why_match": str,
      "explanation": {{
        "skills": str,
        "experience": str,
        "education": str,
        "aspirations": str
      }}
    }}
  ],
  "suggestions": [
    {{
      "skill_or_experience": str,
      "suggestion": str
    }}
  ]
}}
"""
    ),
    ("system", "Context (job postings):\n{context}"),
    ("human", "Match jobs for CV with input:\n{input}")
])



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
