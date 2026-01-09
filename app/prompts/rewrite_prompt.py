"""
Rewrite prompt for CV query reformulation.
Used in rag_matching.get_rag_components().
"""

from langchain_core.prompts import ChatPromptTemplate

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rewrite CV info into a concise job search query."),
    ("human", "{input}")
])