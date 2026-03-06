"""
Reverse matching: Find candidates for a job using RAG on CV index.
"""

import json
import logging
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from models.core import CandidateSearchInput, CandidateSearchResponse, MatchedCandidate, Suggestion
from services.api_key_manager import get_next_api_key
from services.chroma_utils import get_vectorstore

from prompts import qa_prompt
from services.db_utils import get_db_connection
from services.rag_helpers import _prefix_doc_with_id  # Reuse QA prompt, adjust for reverse

logging.basicConfig(level=logging.INFO)

def get_candidate_rag_components(model="gemini-2.5-flash"):
    """Get RAG for candidate search (job query → CV docs)."""
    google_api_key = get_next_api_key()
    llm = ChatGoogleGenerativeAI(model=model, google_api_key=google_api_key, temperature=0.2)
    
    # CV retriever (assume CVs indexed in separate collection or same with metadata['type']='cv')
    cv_vectorstore = get_vectorstore(collection_name="cv_collection")  # Or filter where type='cv'
    retriever = cv_vectorstore.as_retriever(search_kwargs={"k": 20})
    
    # Reuse QA prompt, but adjust for reverse (job input → CV output)
    reverse_qa_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a candidate matching assistant.
Use ONLY the provided CV context.
Select top 5 most relevant CVs for the job description.

Weights: 50% skills, 30% experience, 10% aspirations, 10% education.

IMPORTANT:
- cv_id MUST come from 'CV_ID:' in context (prefix in docs)
- Do NOT invent CVs

Return STRICT JSON:
{{
  "matched_candidates": [{{
    "cv_id": int,
    "name": str,
    "email": str,
    "phone": str,
    "match_score": float (0-1 scale),
    "matched_skills": [str],
    "matched_experience": [str],
    "matched_education": [str],
    "why_match": str (Vietnamese explanation, max 100 words)
  }}],
  "suggestions": [{{"skill_or_experience": str, "suggestion": str}}]
}}""",
        ),
        ("system", "Context (CV postings):\n{context}"),
        ("human", "Find CVs for job: {input}")
    ])
    
    qa_chain = reverse_qa_prompt | llm | JsonOutputParser()
    return retriever, qa_chain

async def match_candidates_to_job(input: CandidateSearchInput) -> CandidateSearchResponse:
    """Reverse match: Job desc → top CVs."""
    try:
        job_query = input.job_description
        filters = input.filters
        limit = input.limit
        
        retriever, qa_chain = get_candidate_rag_components(input.model) # type: ignore #
        cv_vectorstore = get_vectorstore(collection_name="cv_collection")
        
        # Apply filters (e.g., exp_years >2)
        where_filter = {}
        if 'experience_years' in filters:
            where_filter['exp_years'] = {"$gte": filters['experience_years']}
        if 'skills' in filters:
            where_filter['skills'] = {"$in": filters['skills']}
        
        # Retrieve CV docs
        cv_docs = cv_vectorstore.similarity_search(job_query, k=20, filter=where_filter)
        cv_docs = [_prefix_doc_with_id(d, prefix="CV_ID") for d in cv_docs]  # type: ignore # Prefix CV_ID for prompt
        
        logging.info(f"Retrieved {len(cv_docs)} CVs for job query")
        
        # QA chain for ranking
        result = await qa_chain.ainvoke({
            "context": cv_docs,
            "input": job_query
        })
        
        matched_cvs = result.get("matched_candidates", [])
        suggestions = result.get("suggestions", [])
        
        # Enrich with full CV info from DB
        enriched_cvs = []
        for cv in matched_cvs:
            cv_id = cv['cv_id']
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT cv_info_json FROM cv_store WHERE id = ?", (cv_id,))
                row = cursor.fetchone()
                if row:
                    full_cv = json.loads(row['cv_info_json'])
                    enriched_cv = MatchedCandidate(
                        cv_id=cv_id,
                        name=full_cv.get('name', 'N/A'),
                        email=full_cv.get('email', 'N/A'),
                        phone=full_cv.get('phone', 'N/A'),
                        match_score=cv['match_score'],
                        matched_skills=cv['matched_skills'],
                        matched_experience=cv['matched_experience'],
                        matched_education=cv['matched_education'],
                        career_objective=full_cv.get('career_objective', ''),
                        education=full_cv.get('education', []),
                        experience=full_cv.get('experience', []),
                        why_match=cv['why_match'],
                        cv_info_summary=full_cv
                    )
                    enriched_cvs.append(enriched_cv)
        
        # Sort by score
        enriched_cvs.sort(key=lambda x: x.match_score, reverse=True)
        
        logging.info(f"Matched {len(enriched_cvs)} CVs for job")
        return CandidateSearchResponse(
            total=len(enriched_cvs),
            candidates=enriched_cvs[:limit],  # Limit results
            suggestions=suggestions
        )
        
    except Exception as e:
        logging.error(f"Reverse match failed: {e}")
        return CandidateSearchResponse(
            total=0,
            candidates=[],
            suggestions=[Suggestion(skill_or_experience="N/A", suggestion=str(e))]
        )