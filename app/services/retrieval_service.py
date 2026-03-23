# app/services/retrieval_service.py
"""
Retrieval Service - RAG retrieval logic from ChromaDB.
Handles semantic search, filtering, dan document retrieval.
"""

import logging
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document

from .chroma_utils import get_vectorstore
from app.models.chatbot import RetrievedDocument

logger = logging.getLogger(__name__)


class RetrievalService:
    """Service cho RAG retrieval operations"""
    
    def __init__(self, collection_name: str = "jobs", k: int = 3):
        """
        Initialize Retrieval Service
        
        Args:
            collection_name: ChromaDB collection name (jobs, cvs)
            k: Number of documents to retrieve
        """
        self.collection_name = collection_name
        self.k = k
        self.vectorstore = get_vectorstore(collection_name)
    
    def retrieve(
        self, 
        query: str, 
        k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedDocument]:
        """
        Retrieve documents matching the query.
        Includes retry logic for 429 rate limit errors.
        """
        import time
        k = k or self.k
        logger.info(f" Retrieving {k} docs for query: {query[:50]}...")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                results = self.vectorstore.similarity_search_with_relevance_scores(
                    query, 
                    k=k
                )
                
                retrieved_docs = []
                for doc, score in results:
                    retrieved_docs.append(
                        RetrievedDocument(
                            id=str(doc.metadata.get("job_id", "unknown")),
                            content=doc.page_content,
                            metadata=doc.metadata or {},
                            distance=score
                        )
                    )
                
                logger.info(f" Retrieved {len(retrieved_docs)} documents")
                return retrieved_docs
                
            except Exception as e:
                err_str = str(e)
                if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str) and attempt < max_retries - 1:
                    wait = 20 * (attempt + 1)
                    logger.warning(f" Rate limited (attempt {attempt+1}), retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                logger.error(f" Retrieval failed: {e}")
                return []
        
        return []
    
    def retrieve_by_metadata(
        self, 
        field: str, 
        value: str,
        k: int = 10
    ) -> List[RetrievedDocument]:
        """
        Retrieve documents by metadata field
        
        Args:
            field: Metadata field name (e.g., "job_title", "location")
            value: Field value to match
            k: Number of results
            
        Returns:
            List of matching documents
        """
        try:
            logger.info(f" Searching {field}={value}")
            
            # This is a simple implementation
            # For more complex filtering, you'd need to use Chroma's native filtering
            docs = self.vectorstore.get(
                where={field: value},
                limit=k
            )
            
            retrieved_docs = []
            for doc_id, metadata in zip(docs.get('ids', []), docs.get('metadatas', [])):
                retrieved_docs.append(
                    RetrievedDocument(
                        id=str(doc_id),
                        content=metadata.get('content', ''),
                        metadata=metadata,
                        distance=None
                    )
                )
            
            return retrieved_docs
            
        except Exception as e:
            logger.error(f" Metadata search failed: {e}")
            return []
    
    def get_context_string(
        self, 
        query: str, 
        k: Optional[int] = None
    ) -> str:
        """
        Get retrieved documents as formatted context string
        
        Args:
            query: Search query
            k: Number of documents
            
        Returns:
            Formatted context string for LLM
        """
        docs = self.retrieve(query, k=k)
        
        if not docs:
            return "Không tìm thấy tài liệu phù hợp."
        
        context_parts = []
        for i, doc in enumerate(docs, 1):
            title = doc.metadata.get("job_title", "Job")
            location = doc.metadata.get("location", "N/A")
            company = doc.metadata.get("company", "N/A")
            
            context_parts.append(f"""
[Document {i}]
Chức vị: {title}
Công ty: {company}
Địa điểm: {location}
Chi tiết:
{doc.content[:300]}...
---
""")
        
        return "\n".join(context_parts)
    
    def retrieve_similar_jobs(
        self, 
        job_id: int, 
        k: int = 5
    ) -> List[RetrievedDocument]:
        """
        Find similar jobs to a given job
        
        Args:
            job_id: Reference job ID
            k: Number of similar jobs to return
            
        Returns:
            List of similar documents (excluding the reference job)
        """
        try:
            logger.info(f" Finding similar jobs to job_id={job_id}")
            
            # Get the reference job details
            all_docs = self.vectorstore.get(limit=1000)
            
            reference_job = None
            for metadata in all_docs.get('metadatas', []):
                if str(metadata.get('job_id')) == str(job_id):
                    reference_job = metadata
                    break
            
            if not reference_job:
                logger.warning(f" Job {job_id} not found")
                return []
            
            # Use job title as query
            query = f"{reference_job.get('job_title')} {reference_job.get('company', '')}"
            similar_docs = self.retrieve(query, k=k+1)  # +1 to account for self
            
            # Filter out the reference job itself
            similar_docs = [
                doc for doc in similar_docs 
                if doc.id != str(job_id)
            ][:k]
            
            return similar_docs
            
        except Exception as e:
            logger.error(f" Similar jobs search failed: {e}")
            return []
    
    def hybrid_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        k: Optional[int] = None
    ) -> List[RetrievedDocument]:
        """
        Hybrid search: semantic similarity + ChromaDB metadata filters.
        
        Args:
            query: Search text for semantic matching
            filters: Dict of metadata filters, e.g. {"work_type": "remote"}
            k: Number of results
        """
        import time
        k = k or self.k
        logger.info(f" Hybrid search: query={query[:50]}..., filters={filters}")

        # Build ChromaDB where clause from filters
        where = None
        if filters:
            conditions = []
            for key, value in filters.items():
                if value:
                    conditions.append({key: value})
            if len(conditions) == 1:
                where = conditions[0]
            elif len(conditions) > 1:
                where = {"$and": conditions}

        max_retries = 3
        for attempt in range(max_retries):
            try:
                results = self.vectorstore.similarity_search_with_relevance_scores(
                    query, k=k, filter=where
                )

                retrieved_docs = []
                for doc, score in results:
                    retrieved_docs.append(
                        RetrievedDocument(
                            id=str(doc.metadata.get("job_id", "unknown")),
                            content=doc.page_content,
                            metadata=doc.metadata or {},
                            distance=score
                        )
                    )

                logger.info(f" Hybrid search returned {len(retrieved_docs)} docs")
                return retrieved_docs

            except Exception as e:
                err_str = str(e)
                if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str) and attempt < max_retries - 1:
                    wait = 20 * (attempt + 1)
                    logger.warning(f" Rate limited (attempt {attempt+1}), retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                # If filter caused the error, retry without filter
                if where and attempt == 0:
                    logger.warning(f" Hybrid search with filter failed, retrying without filter: {e}")
                    where = None
                    continue
                logger.error(f" Hybrid search failed: {e}")
                return []

        return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Compute aggregate stats from ChromaDB metadata.
        Replaces PostgreSQL get_job_stats() for chat context.
        """
        from collections import Counter
        try:
            all_data = self.vectorstore._collection.get(
                include=["metadatas"],
                limit=10000
            )
            metadatas = all_data.get("metadatas") or []
            total = len(metadatas)

            companies = set()
            skill_counter: Counter = Counter()
            cat_counter: Counter = Counter()
            location_counter: Counter = Counter()
            work_type_counter: Counter = Counter()

            for m in metadatas:
                if not m:
                    continue
                company = m.get("company", "")
                if company:
                    companies.add(company)

                skills_val = m.get("skills", "")
                for s in str(skills_val).split(","):
                    s = s.strip()
                    if s:
                        skill_counter[s] += 1

                cat = m.get("category", "")
                if cat:
                    cat_counter[cat] += 1

                loc = m.get("company_city", "") or m.get("location", "")
                if loc:
                    location_counter[loc] += 1

                wt = m.get("work_type", "")
                if wt:
                    work_type_counter[wt] += 1

            return {
                "total_jobs": total,
                "total_companies": len(companies),
                "top_categories": [{"name": n, "count": c} for n, c in cat_counter.most_common(10)],
                "top_skills": [{"name": n, "count": c} for n, c in skill_counter.most_common(10)],
                "top_locations": [{"name": n, "count": c} for n, c in location_counter.most_common(10)],
                "work_type_dist": [{"name": n, "count": c} for n, c in work_type_counter.most_common()],
            }

        except Exception as e:
            logger.error(f" Collection stats failed: {e}")
            return {"total_jobs": 0, "total_companies": 0, "top_categories": [], "top_skills": []}

    def aggregate_search(
        self,
        text_contains: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        limit: int = 5000
    ) -> Dict[str, Any]:
        """
        Get matching documents for aggregation (counting, grouping).
        Uses ChromaDB get() instead of similarity search (no embedding needed).
        """
        try:
            kwargs: Dict[str, Any] = {"include": ["metadatas"], "limit": limit}

            if metadata_filter:
                conditions = []
                for key, value in metadata_filter.items():
                    if value:
                        conditions.append({key: value})
                if len(conditions) == 1:
                    kwargs["where"] = conditions[0]
                elif len(conditions) > 1:
                    kwargs["where"] = {"$and": conditions}

            if text_contains:
                kwargs["where_document"] = {"$contains": text_contains}

            results = self.vectorstore._collection.get(**kwargs)

            return {
                "count": len(results.get("ids", [])),
                "metadatas": results.get("metadatas", []),
                "ids": results.get("ids", [])
            }
        except Exception as e:
            logger.error(f" Aggregate search failed: {e}")
            return {"count": 0, "metadatas": [], "ids": []}

    def change_collection(self, collection_name: str):
        """Switch to different ChromaDB collection"""
        self.collection_name = collection_name
        self.vectorstore = get_vectorstore(collection_name)
        logger.info(f"switched to collection: {collection_name}")


# Global Retrieval Service instances
_job_retrieval_service: Optional[RetrievalService] = None
_cv_retrieval_service: Optional[RetrievalService] = None


def get_retrieval_service(collection_name: str = "jobs") -> RetrievalService:
    """Get or create retrieval service instance"""
    global _job_retrieval_service, _cv_retrieval_service
    
    if collection_name == "cvs":
        if _cv_retrieval_service is None:
            _cv_retrieval_service = RetrievalService(collection_name="cvs", k=3)
        return _cv_retrieval_service
    else:
        if _job_retrieval_service is None:
            _job_retrieval_service = RetrievalService(collection_name="jobs", k=3)
        return _job_retrieval_service


def reset_retrieval_services():
    """Reset all retrieval services (for testing)"""
    global _job_retrieval_service, _cv_retrieval_service
    _job_retrieval_service = None
    _cv_retrieval_service = None
