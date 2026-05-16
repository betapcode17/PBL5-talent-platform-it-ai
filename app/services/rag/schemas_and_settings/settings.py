"""RAG settings moved under schemas_and_settings package."""

from __future__ import annotations

import os
from pathlib import Path


RAG_COLLECTION_NAME = os.getenv("RAG_COLLECTION_NAME", "jobs_rag")
RAG_BASE_DIR = Path(__file__).resolve().parents[3]
RAG_VECTOR_DIR = os.getenv("RAG_VECTOR_DIR", str(RAG_BASE_DIR / ".rag_store" / "chroma" / RAG_COLLECTION_NAME))
CHROMA_COLLECTION_NAME = RAG_COLLECTION_NAME
CHROMA_PERSIST_DIRECTORY = RAG_VECTOR_DIR
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "8"))
RAG_CANDIDATE_K = int(os.getenv("RAG_CANDIDATE_K", "24"))
RAG_CACHE_TTL_SECONDS = int(os.getenv("RAG_CACHE_TTL_SECONDS", "45"))
RAG_SYNC_INTERVAL_SECONDS = int(os.getenv("RAG_SYNC_INTERVAL_SECONDS", "90"))
RAG_BACKEND_PAGE_SIZE = int(os.getenv("RAG_BACKEND_PAGE_SIZE", "100"))
RAG_BACKEND_MAX_PAGES = int(os.getenv("RAG_BACKEND_MAX_PAGES", "200"))
RAG_COMPANY_MAX_PAGES = int(os.getenv("RAG_COMPANY_MAX_PAGES", "200"))
RAG_CHUNK_SIZE_CHARS = int(os.getenv("RAG_CHUNK_SIZE_CHARS", "900"))
RAG_CHUNK_OVERLAP_CHARS = int(os.getenv("RAG_CHUNK_OVERLAP_CHARS", "160"))
RAG_CHUNK_MAX_TOKENS = int(os.getenv("RAG_CHUNK_MAX_TOKENS", "320"))
RAG_MAX_CHUNKS_PER_SOURCE = int(os.getenv("RAG_MAX_CHUNKS_PER_SOURCE", "2"))
RAG_REBUILD_ON_SYNC = os.getenv("RAG_REBUILD_ON_SYNC", "true").lower() == "true"

# Retrieval scoring weights - customize via environment or presets
RAG_RERANK_PRESET = os.getenv("RAG_RERANK_PRESET", "balanced")
RAG_SEMANTIC_WEIGHT = float(os.getenv("RAG_SEMANTIC_WEIGHT", "0.35"))
RAG_BM25_WEIGHT = float(os.getenv("RAG_BM25_WEIGHT", "0.30"))
RAG_TITLE_WEIGHT = float(os.getenv("RAG_TITLE_WEIGHT", "0.12"))
RAG_COMPANY_WEIGHT = float(os.getenv("RAG_COMPANY_WEIGHT", "0.08"))
RAG_SKILLS_WEIGHT = float(os.getenv("RAG_SKILLS_WEIGHT", "0.08"))
RAG_CATEGORY_WEIGHT = float(os.getenv("RAG_CATEGORY_WEIGHT", "0.05"))
RAG_ENTITY_BIAS_WEIGHT = float(os.getenv("RAG_ENTITY_BIAS_WEIGHT", "0.02"))
RAG_USE_CUSTOM_WEIGHTS = os.getenv("RAG_USE_CUSTOM_WEIGHTS", "false").lower() == "true"

# Cross-encoder reranking
RAG_USE_CROSS_ENCODER = os.getenv("RAG_USE_CROSS_ENCODER", "false").lower() == "true"
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-12-v2")
RAG_CROSS_ENCODER_THRESHOLD = float(os.getenv("RAG_CROSS_ENCODER_THRESHOLD", "0.5"))

# Full-text search hybrid mode
RAG_USE_FULLTEXT_SEARCH = os.getenv("RAG_USE_FULLTEXT_SEARCH", "false").lower() == "true"
RAG_FULLTEXT_DB_PATH = os.getenv("RAG_FULLTEXT_DB_PATH", str(RAG_BASE_DIR / ".rag_store" / "fulltext.db"))
FULLTEXT_DB_PATH = RAG_FULLTEXT_DB_PATH
RAG_FULLTEXT_WEIGHT = float(os.getenv("RAG_FULLTEXT_WEIGHT", "0.10"))
RAG_FULLTEXT_TOP_K = int(os.getenv("RAG_FULLTEXT_TOP_K", "24"))
RAG_DEFAULT_PROFILE = os.getenv("RAG_DEFAULT_PROFILE", "balanced")
RAG_CONTEXT_MAX_TOKENS = int(os.getenv("RAG_CONTEXT_MAX_TOKENS", "1800"))
RAG_MAX_CHUNK_TOKENS = int(os.getenv("RAG_MAX_CHUNK_TOKENS", "320"))
RAG_DEDUPE_SIMILARITY_THRESHOLD = float(os.getenv("RAG_DEDUPE_SIMILARITY_THRESHOLD", "0.92"))
RAG_LOW_CONFIDENCE_TOP_SCORE = float(os.getenv("RAG_LOW_CONFIDENCE_TOP_SCORE", "0.42"))
RAG_LOW_CONFIDENCE_MIN_ITEMS = int(os.getenv("RAG_LOW_CONFIDENCE_MIN_ITEMS", "2"))
RAG_RETRIEVAL_RETRY_COUNT = int(os.getenv("RAG_RETRIEVAL_RETRY_COUNT", "1"))
RAG_RETRIEVAL_TIMEOUT_MS = int(os.getenv("RAG_RETRIEVAL_TIMEOUT_MS", "8000"))
RAG_NORMALIZATION_STRATEGY = os.getenv("RAG_NORMALIZATION_STRATEGY", "min-max")
RAG_ENABLE_BM25 = os.getenv("RAG_ENABLE_BM25", "true").lower() == "true"
RAG_ENABLE_SEMANTIC = os.getenv("RAG_ENABLE_SEMANTIC", "true").lower() == "true"
RAG_ENABLE_FALLBACK = os.getenv("RAG_ENABLE_FALLBACK", "true").lower() == "true"
RAG_DEFAULT_FALLBACK_SKILL = os.getenv("RAG_DEFAULT_FALLBACK_SKILL", "general")

PROMPT_MAX_CONTEXT_TOKENS = int(os.getenv("PROMPT_MAX_CONTEXT_TOKENS", str(RAG_CONTEXT_MAX_TOKENS)))
PROMPT_TEMPLATE = os.getenv(
	"PROMPT_TEMPLATE",
	"You are a helpful job-search assistant. Use the provided context to answer the user.\n\nQuestion: {query}\n\nContext:\n{context}\n",
)

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "auto").lower()

QWEN_MODEL_NAME = os.getenv("QWEN_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
QWEN_MAX_NEW_TOKENS = int(os.getenv("QWEN_MAX_NEW_TOKENS", "360"))
QWEN_TEMPERATURE = float(os.getenv("QWEN_TEMPERATURE", "0.2"))
QWEN_TOP_P = float(os.getenv("QWEN_TOP_P", "0.9"))
QWEN_BATCH_SIZE = int(os.getenv("QWEN_BATCH_SIZE", "4"))
QWEN_ENFORCE_CUDA = os.getenv("QWEN_ENFORCE_CUDA", "true").lower() == "true"
QWEN_USE_FLASH_ATTENTION = os.getenv("QWEN_USE_FLASH_ATTENTION", "false").lower() == "true"
QWEN_MAX_INPUT_TOKENS = int(os.getenv("QWEN_MAX_INPUT_TOKENS", "3800"))
QWEN_TORCH_DTYPE = os.getenv("QWEN_TORCH_DTYPE", "float16")
QWEN_USE_8BIT_QUANTIZATION = os.getenv("QWEN_USE_8BIT_QUANTIZATION", "false").lower() == "true"
HF_LOCAL_FILES_ONLY = os.getenv("HF_LOCAL_FILES_ONLY", "false").lower() == "true"

CROSS_ENCODER_BATCH_SIZE = int(os.getenv("CROSS_ENCODER_BATCH_SIZE", "16"))
