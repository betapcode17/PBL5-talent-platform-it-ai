# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
from app.routers import cv, jobs, matching, utils
from app.services.chroma_utils import preload_jobs as preload_jobs_to_chroma
from app.config import DATA_PATH

logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = FastAPI(title="AI CV-Job Matcher", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(cv.router, prefix="/cv", tags=["CV"])
app.include_router(jobs.router, prefix="/jobs", tags=["Jobs"])
app.include_router(matching.router, prefix="/matching", tags=["Matching"])
app.include_router(utils.router, tags=["Utils"])

@app.on_event("startup")
async def startup_event():
    try:
        logging.info("🔄 Preloading jobs into Chroma and SQLite...")
        preload_jobs_to_chroma(DATA_PATH, batch_size=500)
        logging.info("✅ Preloading completed")
    except Exception as e:
        logging.error(f"Error during startup preload: {str(e)}")
        raise

@app.get("/")
async def root():
    return {"message": "CV Matching API is running!"}