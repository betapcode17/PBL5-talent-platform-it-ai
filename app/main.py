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
from app.services.chroma_utils import preload_jobs as preload_jobs_to_chroma, preload_jobs_from_pg

# Import DATA_PATH from config
from app.config import JOBS_CSV_PATH as DATA_PATH

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
    import asyncio
    print(" STARTUP EVENT TRIGGERED")

    async def _preload_in_background():
        """Run preload in background so server starts immediately."""
        try:
            logging.info("Background: Attempting to preload jobs from PostgreSQL...")
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(None, preload_jobs_from_pg)
            if success:
                logging.info(" Background preload from PostgreSQL completed successfully")
            else:
                logging.warning(" Background PostgreSQL preload failed, trying CSV...")
                success = await loop.run_in_executor(
                    None, lambda: preload_jobs_to_chroma(DATA_PATH, batch_size=500) # type: ignore
                )
                if success:
                    logging.info(" Background CSV preload completed")
                else:
                    logging.warning(" Background preload failed entirely")
        except Exception as e:
            logging.exception(f" Background preload error: {e}")

    # Launch preload in background - server starts immediately
    asyncio.create_task(_preload_in_background())
    logging.info(" Server started - job preload running in background")

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

