# AI RAG Service (Qwen CUDA)

RAG-only FastAPI service for job retrieval and conversational response generation.

## What this module does
- Fetches job data from backend API.
- Chunks and embeds data with sentence-transformers (BGE).
- Stores vectors in ChromaDB.
- Retrieves relevant context per user query.
- Injects context into prompt and generates answer with Qwen on GPU.

## Core architecture
- Ingestion: app/services/rag/ingestion.py
- Embedding: app/services/rag/embedding.py
- Vector store: app/services/rag/vector_store.py
- Retrieval: app/services/rag/retrieval.py
- Generation: app/services/rag/generation.py
- Pipeline orchestration: app/services/rag/pipeline.py
- Chat endpoint integration: app/routers/chatbot.py

## Start
1. Install dependencies.
2. Configure .env.
3. Run server:

python run_server.py

## Important env vars
- QWEN_MODEL_NAME
- QWEN_ENFORCE_CUDA
- EMBED_MODEL_NAME
- RAG_COLLECTION_NAME
- RAG_SYNC_INTERVAL_SECONDS
- BACKEND_API_URL

## Maintenance scripts
- scripts/load_jobs_from_backend.py
- scripts/health_check.py

## API docs
- /docs
- /redoc
