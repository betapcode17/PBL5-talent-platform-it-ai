# app/main.py
"""
Main FastAPI application entry point for AI CV-Job Matcher.
Handles startup (preload jobs), middleware, and router inclusion.
"""

print(" main.py loaded")
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Calculate paths and load .env from root
BASE_DIR = Path(__file__).resolve().parent  # app/
PROJECT_ROOT = BASE_DIR.parent              # root (ai-cv-job-matcher/)
load_dotenv(PROJECT_ROOT / ".env")

# Logging config (file + console for dev)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(PROJECT_ROOT / "app.log"),  # Log to root/app.log
        logging.StreamHandler()  # Also print to console
    ]
)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Import routers with app. prefix (absolute from root)
from app.routers.cv import router as cv_router
from app.routers.jobs import router as jobs_router
from app.routers.matching import router as matching_router
from app.routers.utils import router as utils_router
from app.routers.chatbot import router as chatbot_router
from app.routers import candidates

# Import error handlers
from app.middleware.error_handler import setup_error_handlers
# Import preload function
from app.services.chatbot_service import get_chatbot

app = FastAPI(
    title="AI CV-Job Matcher",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(cv_router, prefix="/cv", tags=["CV"])
app.include_router(jobs_router, prefix="/jobs", tags=["Jobs"])
app.include_router(matching_router, prefix="/matching", tags=["Matching"])
app.include_router(chatbot_router)  # Chatbot router doesn't need prefix (has /chatbot already)
app.include_router(utils_router, tags=["Utils"])
app.include_router(candidates.router, prefix="/candidates", tags=["Candidates"])

# Mount static files
static_path = PROJECT_ROOT / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
    logging.info(f" Static files mounted from {static_path}")
else:
    logging.warning(f" Static directory not found: {static_path}")

# Setup error handlers
setup_error_handlers(app)
logging.info(" Error handlers setup complete")

@app.on_event("startup")
async def startup_event():
    print(" STARTUP EVENT TRIGGERED")
    
    import asyncio
    
    # ==========================
    # PRELOAD MODELS ON STARTUP
    # ==========================
    async def _preload_models():
        """Preload all ML models (embeddings + Qwen) on server startup."""
        try:
            logging.info(" Preloading ML models...")
            
            # Get RAG pipeline (triggers lazy-load of embedding + generation models)
            pipeline = get_chatbot().pipeline
            
            # Trigger embedding model load
            _ = pipeline.embedding.model  # Access .model property to trigger load
            logging.info(" ✓ Embedding model preloaded")
            
            # Trigger Qwen model load (already in __init__, but verify)
            logging.info(" ✓ Qwen generation model preloaded")
            
            # Health check
            health = pipeline.health()
            logging.info(" ✓ Models ready - health check: %s", health)
            
        except Exception as e:
            logging.exception(f" Model preload error (non-fatal): {e}")
            logging.warning(" Server will attempt lazy-load models on first request")
    
    # Preload models before proceeding
    await _preload_models()

    if os.getenv("ENABLE_RAG_WARMUP", "0") != "1":
        logging.info(" RAG warmup skipped; set ENABLE_RAG_WARMUP=1 to enable background sync")
        return

    async def _sync_rag_in_background():
        """Warm up RAG index in background while server starts."""
        try:
            chatbot = get_chatbot()
            result = await chatbot.force_sync()
            logging.info("Background RAG sync complete: %s", result)
        except Exception as e:
            logging.exception(f" Background RAG sync error: {e}")

    # Launch sync in background - server starts immediately
    asyncio.create_task(_sync_rag_in_background())
    logging.info(" Server started - RAG sync running in background")

# =========================
# ROOT ENDPOINT
# =========================
# uvicorn main:app --reload
@app.get("/")
async def root():
    return {"message": "CV Matching API is running!"}

@app.get("/chat")
async def chat_page():
    """Serve chat UI"""
    try:
        with open(PROJECT_ROOT / "static" / "index.html") as f:
            from fastapi.responses import HTMLResponse
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return {
            "message": "Chat UI not found. Please run 'python scripts/preload_embeddings.py' first",
            "api_docs": "/docs"
        }

