# app/config.py
"""
Configuration for AI CV-Job Matcher application.
Loads settings from .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
BASE_DIR = Path(__file__).resolve().parent  # app/
PROJECT_ROOT = BASE_DIR.parent
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE)

# Paths
BASE_DIR = Path(__file__).resolve().parent  # app/
PROJECT_ROOT = BASE_DIR.parent

# CSV path (legacy, kept for backward compatibility)
JOBS_CSV_PATH = BASE_DIR / "data" / "jobs_vietnamese.csv"

# LLM Provider Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # Ollama only (local LLM)

# Ollama Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

# Database Configuration - PostgreSQL for chatbot persistence
# DATABASE_URL is shared by chatbot storage and any other relational services.
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:17122005@localhost:5432/chat_bot"
)

# Backend API Configuration (NestJS)
BACKEND_API_URL = os.getenv(
    "BACKEND_API_URL",
    "http://localhost:4000"
)
