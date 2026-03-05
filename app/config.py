# app/config.py
"""
Configuration for AI CV-Job Matcher application.
"""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent  # app/
PROJECT_ROOT = BASE_DIR.parent

# CSV path (legacy, kept for backward compatibility)
JOBS_CSV_PATH = BASE_DIR / "data" / "jobs_vietnamese.csv"

# Google API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# PostgreSQL
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:17122005@localhost:5432/it_job_db"
)
