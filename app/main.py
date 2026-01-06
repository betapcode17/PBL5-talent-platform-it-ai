# app/main.py
"""
Main FastAPI application entry point for AI CV-Job Matcher.
Handles startup (preload jobs), middleware, and router inclusion.
"""

print("✅ main.py loaded")
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

# Import routers with app. prefix (absolute from root)
from routers.cv import router as cv_router
from routers.jobs import router as jobs_router
from routers.matching import router as matching_router
from routers.utils import router as utils_router

# Import preload function
from services.chroma_utils import preload_jobs as preload_jobs_to_chroma

# Import DATA_PATH from config
from config import JOBS_CSV_PATH as DATA_PATH

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
app.include_router(utils_router, tags=["Utils"])

# @app.on_event("startup")
# async def startup_event():
#     print("🔄 STARTUP EVENT TRIGGERED")
#     try:
#         logging.info(f"Preloading jobs from {DATA_PATH} into Chroma and SQLite...")
#         success = preload_jobs_to_chroma(DATA_PATH, batch_size=500)
#         if success:
#             logging.info("✅ Preloading completed successfully")
#         else:
#             logging.warning("⚠️ Preloading skipped or failed (check logs)")
#     except Exception as e:
#         logging.exception(f"❌ Error during startup preload: {e}")
#         raise  # Re-raise to prevent app start if critical

# =========================
# ROOT ENDPOINT
# =========================
@app.get("/")
async def root():
    return {"message": "CV Matching API is running!"}