import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "jobs_processed.jsonl")
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "db", "cv_job_matching.db")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not set in .env")