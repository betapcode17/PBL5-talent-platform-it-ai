"""
Business logic services for CV-Job matching.
Includes AI, DB, Chroma, and RAG services.
"""

from .ai_analysis import (
    analyze_cv_insights,
    generate_cv_improvements,
    generate_why_match,
    generate_question_suggestions,
    get_llm
)
from .api_key_manager import get_next_api_key, get_api_key_manager
from .chroma_utils import (
    get_vectorstore,
    preload_jobs,
    index_cv_extracts,
    delete_cv_from_chroma
)
from .db_utils import (
    get_db_connection,
    create_tables,
    insert_cv_record,
    insert_match_log,
    get_cached_matches,
    get_match_history,
    get_all_cvs,
    delete_cv_record,
    get_filtered_jobs,
    get_total_jobs,
    insert_application,
    get_applications_by_cv,
    check_application_exists,
    save_cv_insights,
    get_cv_insights,
    save_document_preview,
    get_document_preview
)
from .rag_matching import match_cv, get_rag_components
from .rag_helpers import _to_int_job_id, _prefix_doc_with_id, verify_job_id_consistency

__all__ = [
    # AI Analysis
    'analyze_cv_insights', 'generate_cv_improvements', 'generate_why_match',
    'generate_question_suggestions', 'get_llm',
    # API Key
    'get_next_api_key', 'get_api_key_manager',
    # Chroma
    'get_vectorstore', 'preload_jobs', 'index_cv_extracts', 'delete_cv_from_chroma',
    # DB Utils
    'get_db_connection', 'create_tables', 'insert_cv_record', 'insert_match_log',
    'get_cached_matches', 'get_match_history', 'get_all_cvs', 'delete_cv_record',
    'get_filtered_jobs', 'get_total_jobs', 'insert_application', 'get_applications_by_cv',
    'check_application_exists', 'save_cv_insights', 'get_cv_insights',
    'save_document_preview', 'get_document_preview',
    # RAG
    'match_cv', 'get_rag_components',
    # Helpers
    '_to_int_job_id', '_prefix_doc_with_id', 'verify_job_id_consistency'
]