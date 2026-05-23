# app/services/db_utils.py
"""Minimal DB utilities shim used during development and tests.
This file provides simple placeholder implementations so the routers can
import the expected functions. In production these should be replaced
with the real DB integration.
"""
from contextlib import contextmanager
import sqlite3
from typing import List, Dict, Any, Optional

@contextmanager
def get_db_connection():
    # ephemeral in-memory sqlite for compatibility; callers may monkeypatch
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

# Placeholder implementations
def get_all_cvs() -> List[Dict[str, Any]]:
    return []

def get_cached_matches() -> List[Dict[str, Any]]:
    return []

def get_filtered_jobs(filters: Dict[str, Any]) -> List[int]:
    return []

def get_jobs_details_by_ids(ids: List[int]) -> List[Dict[str, Any]]:
    return []

def insert_cv_record(filename: str, cv_info: Dict[str, Any], file_data: bytes) -> Optional[int]:
    return 1

def insert_match_log(cv_id: int, job_matches: List[Dict[str, Any]], session_id: str) -> None:
    return None

def insert_application(cv_id: int, job_id: int, cover_letter: str, status: str) -> int:
    return 1

def get_applications_by_cv(cv_id: int, status: Optional[str] = None) -> List[Dict[str, Any]]:
    return []

def check_application_exists(cv_id: int, job_id: int) -> bool:
    return False
