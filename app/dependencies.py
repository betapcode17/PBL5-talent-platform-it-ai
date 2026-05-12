def get_db():
    from app.services.db_utils import get_db_connection

    return get_db_connection()

def get_llm_instance():
    from app.services.ai_analysis import get_llm

    return get_llm()

def get_vector_store():
    from app.services.chroma_utils import get_vectorstore

    return get_vectorstore()