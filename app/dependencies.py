from services.db_utils import get_db_connection
from services.ai_analysis import get_llm
from services.chroma_utils import get_vectorstore
from services.api_key_manager import get_next_api_key

def get_db():
    return get_db_connection()

def get_llm_instance():
    return get_llm()

def get_vector_store():
    return get_vectorstore()

def get_api_key():
    return get_next_api_key()