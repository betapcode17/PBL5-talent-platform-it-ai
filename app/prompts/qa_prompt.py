"""
QA prompt for job matching with JSON output.
Escaped literals for LangChain template safety.
Used in rag_matching.get_rag_components().
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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
""",
    ),
    ("system", "Context (job postings):\n{context}"),
    ("human", "Match jobs for CV with input:\n{input}")
])