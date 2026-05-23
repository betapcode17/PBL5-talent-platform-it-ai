# AI RAG Service

FastAPI service for job retrieval, chatbot responses, and GPU-backed generation.

## Overview

This module powers the AI layer of the system. It ingests job and company data from the backend API, converts it into searchable chunks, stores embeddings in ChromaDB, retrieves the most relevant context for each user query, and generates the final answer with Qwen.

## What it does

- Fetches job and company data from the backend API.
- Chunks and embeds content with sentence-transformers.
- Stores vectors in ChromaDB.
- Runs hybrid retrieval with semantic search, BM25, metadata boosting, and fallback logic.
- Builds a prompt from the best matches and generates the response with Qwen on GPU.
- Exposes chatbot endpoints through FastAPI.

## Architecture

```text
Backend API -> Ingestion -> Chunking -> Embedding -> ChromaDB
										  |
										  v
Query -> Normalization -> Retrieval -> Ranking -> Prompt Builder -> Qwen -> Response
```

### Core modules

- Ingestion: [app/services/rag/core/ingestion.py](app/services/rag/core/ingestion.py)
- Retrieval: [app/services/rag/core/retrieval.py](app/services/rag/core/retrieval.py)
- Pipeline orchestration: [app/services/rag/core/pipeline.py](app/services/rag/core/pipeline.py)
- Embedding: [app/services/rag/infra/embedding.py](app/services/rag/infra/embedding.py)
- Vector store: [app/services/rag/stores/vector_store.py](app/services/rag/stores/vector_store.py)
- Fulltext fallback store: [app/services/rag/stores/fulltext_store.py](app/services/rag/stores/fulltext_store.py)
- Prompt builder: [app/services/rag/prompt_and_response/prompt_builder.py](app/services/rag/prompt_and_response/prompt_builder.py)
- Response formatter: [app/services/rag/prompt_and_response/response_formatter.py](app/services/rag/prompt_and_response/response_formatter.py)
- Chat endpoint integration: [app/routers/chatbot.py](app/routers/chatbot.py)

## Retrieval flow

The retrieval pipeline uses a hybrid approach:

1. Normalize and rewrite the query.
2. Detect job intent, city hints, and role hints.
3. Collect candidates from semantic search and BM25.
4. Apply metadata boosts for title, company, description, skills, and category.
5. Optionally rerank with a cross-encoder.
6. Filter low-relevance or duplicate candidates.
7. Build prompt context from the final candidates.

This design improves recall for short queries like "backend Python HCM" while keeping the final context focused.

## Getting started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Copy or update [AI/.env](.env) with the required values for:

- `BACKEND_API_URL`
- `RAG_COLLECTION_NAME`
- `RAG_SYNC_INTERVAL_SECONDS`
- `EMBED_MODEL_NAME`
- `QWEN_MODEL_NAME`
- `QWEN_ENFORCE_CUDA`

### 3. Run the server

```bash
python run_server.py
```

If you need to run uvicorn directly:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

## Important environment variables

| Variable | Purpose |
|---|---|
| `BACKEND_API_URL` | Backend data source for jobs and companies |
| `RAG_COLLECTION_NAME` | ChromaDB collection name |
| `RAG_SYNC_INTERVAL_SECONDS` | Minimum interval between sync runs |
| `RAG_TOP_K` | Number of final results returned |
| `RAG_CANDIDATE_K` | Number of retrieval candidates before ranking |
| `RAG_USE_CROSS_ENCODER` | Enable cross-encoder reranking |
| `RAG_USE_FULLTEXT_SEARCH` | Enable lexical fulltext fallback |
| `EMBED_MODEL_NAME` | Sentence-transformers embedding model |
| `QWEN_MODEL_NAME` | Generation model name |
| `QWEN_ENFORCE_CUDA` | Require GPU execution for generation |

## API

### Chatbot

- `POST /chatbot/message`
- `GET /chatbot/history/{conversation_id}`
- `DELETE /chatbot/conversation/{conversation_id}`

### Docs

- Swagger UI: `/docs`
- ReDoc: `/redoc`

## Maintenance scripts

- `scripts/load_jobs_from_backend.py` - manual ingestion from the backend.
- `scripts/health_check.py` - verify service health and dependencies.

## Notes

- The module is optimized for job-search retrieval, not general-purpose chat.
- Hybrid retrieval is designed to balance semantic understanding with exact keyword matching.
- If retrieval feels too narrow, the first things to inspect are city filtering, relevance gating, and whether fulltext/cross-encoder are enabled.
