"""RAG settings loaded from environment."""

from __future__ import annotations

import os
from pathlib import Path


RAG_COLLECTION_NAME = os.getenv("RAG_COLLECTION_NAME", "jobs_rag")
RAG_BASE_DIR = Path(__file__).resolve().parents[3]
RAG_VECTOR_DIR = os.getenv("RAG_VECTOR_DIR", str(RAG_BASE_DIR / ".rag_store" / "chroma" / RAG_COLLECTION_NAME))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "8"))
RAG_CANDIDATE_K = int(os.getenv("RAG_CANDIDATE_K", "24"))
RAG_CACHE_TTL_SECONDS = int(os.getenv("RAG_CACHE_TTL_SECONDS", "45"))
RAG_SYNC_INTERVAL_SECONDS = int(os.getenv("RAG_SYNC_INTERVAL_SECONDS", "90"))
RAG_BACKEND_PAGE_SIZE = int(os.getenv("RAG_BACKEND_PAGE_SIZE", "100"))
RAG_BACKEND_MAX_PAGES = int(os.getenv("RAG_BACKEND_MAX_PAGES", "200"))
RAG_COMPANY_MAX_PAGES = int(os.getenv("RAG_COMPANY_MAX_PAGES", "200"))
RAG_CHUNK_SIZE_CHARS = int(os.getenv("RAG_CHUNK_SIZE_CHARS", "900"))
RAG_CHUNK_OVERLAP_CHARS = int(os.getenv("RAG_CHUNK_OVERLAP_CHARS", "160"))
RAG_MAX_CHUNKS_PER_SOURCE = int(os.getenv("RAG_MAX_CHUNKS_PER_SOURCE", "2"))
RAG_REBUILD_ON_SYNC = os.getenv("RAG_REBUILD_ON_SYNC", "true").lower() == "true"

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
