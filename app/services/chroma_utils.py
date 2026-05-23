# app/services/chroma_utils.py
"""
Chroma utilities for vector storage and job/CV preloading.
Supports separate collections for jobs and CVs (for reverse matching).
Uses local HuggingFace embeddings (no API key needed).
"""

import os
import json
import logging
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global instances for job and CV collections
_job_vectorstore = None
_cv_vectorstore = None

def get_vectorstore(collection_name: str = "jobs"):
    """
    Khởi tạo Chroma vectorstore với Google Gemini Embedding API.
    Separate collections: "jobs" (default), "cvs" for reverse matching.
    Model: text-embedding-004 (miễn phí, hỗ trợ multilingual).
    """
    global _job_vectorstore, _cv_vectorstore
    
    if collection_name == "cvs":
        if _cv_vectorstore is None:
            _cv_vectorstore = _initialize_vectorstore("cvs")
        return _cv_vectorstore
    else:  # "jobs" default
        if _job_vectorstore is None:
            _job_vectorstore = _initialize_vectorstore("jobs")
        return _job_vectorstore

def _initialize_vectorstore(collection_name: str):
    """Internal init for a specific collection using local Ollama embeddings."""
    try:
        from langchain_chroma import Chroma
        from langchain_ollama import OllamaEmbeddings

        base_dir = Path(__file__).resolve().parent.parent  # app/ -> root
        chroma_path = base_dir / "db" / "chroma_db" / collection_name
        chroma_path.mkdir(parents=True, exist_ok=True)
        
        # Use local Ollama embeddings (no API key needed, no network download)
        # Runs on http://localhost:11434
        from app.config import OLLAMA_BASE_URL
        embedding_function = OllamaEmbeddings(
            base_url=OLLAMA_BASE_URL,
            model="nomic-embed-text"  # Multilingual, lightweight, fast
        )
        
        vectorstore = Chroma(
            persist_directory=str(chroma_path),
            collection_name=collection_name,
            embedding_function=embedding_function
        )
        logging.info(f"✓ Initialized Chroma vectorstore for '{collection_name}' with local Ollama embeddings")
        return vectorstore
    except Exception as e:
        # If initialization fails, try without explicit embedding function (use existing)
        logging.warning(f"⚠ Ollama embeddings init failed for '{collection_name}': {e}")
        logging.info(f"ℹ Attempting to load existing ChromaDB without re-embedding...")
        
        try:
            from langchain_chroma import Chroma

            base_dir = Path(__file__).resolve().parent.parent
            chroma_path = base_dir / "db" / "chroma_db" / collection_name
            
            # Load existing vectorstore without providing embedding function
            vectorstore = Chroma(
                persist_directory=str(chroma_path),
                collection_name=collection_name
            )
            logging.info(f"✓ Loaded existing Chroma vectorstore for '{collection_name}' (using cached embeddings)")
            return vectorstore
        except Exception as e2:
            logging.error(f"✗ Failed to load ChromaDB for '{collection_name}': {e2}")
            # Provide a safe fallback vectorstore that no-ops but preserves API used in code
            class _FallbackCollection:
                def delete(self, where=None):
                    return None
                def count(self):
                    return 0

            class _FallbackVectorStore:
                def __init__(self):
                    self._collection = _FallbackCollection()
                def add_documents(self, docs):
                    logging.debug("FallbackVectorStore.add_documents called - no-op")
                def similarity_search_with_relevance_scores(self, query, k=10):
                    return []
                def similarity_search(self, query, k=10):
                    return []
                def get(self, where=None):
                    return {"ids": []}

            logging.warning(f"Using fallback in-memory vectorstore for '{collection_name}' (Chroma unavailable)")
            return _FallbackVectorStore()

