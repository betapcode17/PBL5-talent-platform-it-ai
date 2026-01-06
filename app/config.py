# app/config.py
import logging
from pathlib import Path
import os
from dotenv import load_dotenv

# ======================
# Stable path resolution (NO cwd dependency)
# ======================
APP_DIR = Path(__file__).resolve().parent          # app/
PROJECT_ROOT = APP_DIR.parent                     # AI/

# ======================
# Load .env from project root
# ======================
load_dotenv(PROJECT_ROOT / ".env")

# ======================
# API KEY
# ======================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not set in .env")

# ======================
# DATA
# ======================
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

JOBS_CSV_PATH = DATA_DIR / "jobs_vietnamese.csv"

# ======================
# DATABASE (SQLite)
# ======================
DB_DIR = APP_DIR / "db"
DB_DIR.mkdir(parents=True, exist_ok=True)

SQLITE_DB_PATH = DB_DIR / "cv_job_matching.db"
SQLALCHEMY_DATABASE_URL = f"sqlite:///{SQLITE_DB_PATH}"

# ======================
# CHROMA
# ======================
CHROMA_PATH = DB_DIR / "chroma_db"
CHROMA_PATH.mkdir(parents=True, exist_ok=True)


# Log paths for debug
logging.info(f"📁 PROJECT_ROOT: {PROJECT_ROOT}")
logging.info(f"📁 APP_DIR: {APP_DIR}")
logging.info(f"📁 DATA_DIR: {DATA_DIR}")
logging.info(f"📁 JOBS_CSV_PATH: {JOBS_CSV_PATH} (exists: {JOBS_CSV_PATH.exists()})")
logging.info(f"📁 DB_DIR: {DB_DIR}")
logging.info(f"📁 CHROMA_PATH: {CHROMA_PATH}")