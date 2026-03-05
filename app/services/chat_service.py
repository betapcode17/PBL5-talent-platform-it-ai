# app/services/chat_service.py
"""
Chat Service - Main chatbot logic combining RAG + LLM.
Handles chat sessions, message processing, and conversation flow.
"""

import logging
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

from services.llm_service import get_llm_service
from services.retrieval_service import get_retrieval_service
from services.pg_database import search_jobs_by_keyword, get_job_stats, get_job_by_id
from models.chat import ChatResponse, ChatHistoryItem, ChatRole, RAGContext
from prompts.chat_system_prompt import CHAT_SYSTEM_PROMPTS

logger = logging.getLogger(__name__)


class ChatbotRAG:
    """Main chatbot class combining RAG + LLM"""
    
    def __init__(
        self, 
        collection_name: str = "jobs",
        k_documents: int = 3,
        enable_rag: bool = True
    ):
        """
        Initialize ChatbotRAG
        
        Args:
            collection_name: ChromaDB collection (jobs, cvs)
            k_documents: Number of docs to retrieve
            enable_rag: Enable RAG retrieval
        """
        self.collection_name = collection_name
        self.k_documents = k_documents
        self.enable_rag = enable_rag
        
        # Initialize services
        self.llm_service = get_llm_service()
        self.retrieval_service = get_retrieval_service(collection_name)
        
        # Chat history storage (in-memory, could use database)
        self.chat_sessions: Dict[str, List[ChatHistoryItem]] = {}
        
        logger.info(f"✅ ChatbotRAG initialized (collection={collection_name})")
    
    def create_session(self) -> str:
        """Create new chat session"""
        session_id = str(uuid.uuid4())
        self.chat_sessions[session_id] = []
        logger.info(f"✅ Created session: {session_id}")
        return session_id
    
    def add_to_history(
        self, 
        session_id: str, 
        role: ChatRole, 
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None
    ):
        """Add message to chat history"""
        if session_id not in self.chat_sessions:
            self.chat_sessions[session_id] = []
        
        message = ChatHistoryItem(
            role=role,
            content=content,
            timestamp=datetime.utcnow(),
            sources=sources
        )
        self.chat_sessions[session_id].append(message)
    
    def get_history(
        self, 
        session_id: str, 
        limit: int = 50
    ) -> List[ChatHistoryItem]:
        """Get chat history for session"""
        if session_id not in self.chat_sessions:
            return []
        
        return self.chat_sessions[session_id][-limit:]
    
    def clear_history(self, session_id: str) -> bool:
        """Clear chat history for session"""
        if session_id in self.chat_sessions:
            self.chat_sessions[session_id] = []
            logger.info(f"✅ Cleared history for session: {session_id}")
            return True
        return False
    
    def _retrieve_context(self, query: str) -> tuple[str, List[Dict[str, Any]]]:
        """
        Retrieve context documents from RAG (ChromaDB) + enrich with PostgreSQL data.
        
        Returns:
            (context_string, sources)
        """
        if not self.enable_rag:
            return "", []
        
        try:
            # 1) Semantic search from ChromaDB
            doc_results = self.retrieval_service.retrieve(
                query, 
                k=self.k_documents
            )
            
            # Format context
            context_parts = []
            sources = []
            
            for doc in doc_results:
                job_id = doc.metadata.get('job_id')
                
                # Enrich with full PostgreSQL data when job_id is available
                pg_job = None
                if job_id and str(job_id).isdigit():
                    try:
                        pg_job = get_job_by_id(int(job_id))
                    except Exception:
                        pass
                
                if pg_job:
                    context_parts.append(f"""
📌 {pg_job.get('job_title', 'Job')}
   Công ty: {pg_job.get('company_name', 'N/A')}
   Địa điểm: {pg_job.get('location', 'N/A')}
   Lương: {pg_job.get('salary_range', 'N/A')}
   Kinh nghiệm: {pg_job.get('experience_required', 'N/A')}
   Kỹ năng: {pg_job.get('skills_text', 'N/A')}
   
{pg_job.get('job_description', '')[:500]}
""")
                else:
                    context_parts.append(f"""
📌 {doc.metadata.get('job_title', 'Job')}
   Công ty: {doc.metadata.get('company', 'N/A')}
   Địa điểm: {doc.metadata.get('location', 'N/A')}
   
{doc.content[:500]}
""")
                
                # Add to sources
                sources.append({
                    "id": doc.id,
                    "title": doc.metadata.get('job_title'),
                    "company": doc.metadata.get('company') or (pg_job.get('company_name') if pg_job else None),
                    "location": doc.metadata.get('location') or (pg_job.get('location') if pg_job else None),
                    "url": doc.metadata.get('url'),
                    "similarity": doc.distance
                })
            
            # 2) Add aggregate stats from PostgreSQL
            try:
                stats = get_job_stats()
                context_parts.append(f"""
📊 Thống kê thị trường:
   Tổng việc làm đang tuyển: {stats.get('total_jobs', 'N/A')}
   Số công ty: {stats.get('total_companies', 'N/A')}
   Top ngành: {', '.join(c['name'] for c in stats.get('top_categories', [])[:5])}
   Top kỹ năng: {', '.join(s['name'] for s in stats.get('top_skills', [])[:5])}
""")
            except Exception:
                pass
            
            context = "\n".join(context_parts) if context_parts else "Không tìm thấy tài liệu."
            return context, sources
            
        except Exception as e:
            logger.error(f"❌ Context retrieval failed: {e}")
            return "", []
    
    def _build_system_prompt(self, context_type: str = "jobs") -> str:
        """Build system prompt for chat"""
        return CHAT_SYSTEM_PROMPTS.get(
            context_type,
            CHAT_SYSTEM_PROMPTS.get("default", "You are a helpful assistant.")
        )
    
    def _build_conversation_context(
        self, 
        session_id: str, 
        max_turns: int = 5
    ) -> str:
        """Build recent conversation for context"""
        history = self.get_history(session_id, limit=max_turns*2)
        
        if not history:
            return ""
        
        context_lines = ["Cuộc trò chuyện trước đây:"]
        for msg in history[-max_turns*2:]:
            role = "👤 Bạn" if msg.role == ChatRole.USER else "🤖 Chatbot"
            context_lines.append(f"{role}: {msg.content[:200]}")
        
        return "\n".join(context_lines)
    
    def chat(
        self,
        user_message: str,
        session_id: Optional[str] = None,
        context_type: str = "jobs",
        use_rag: Optional[bool] = None
    ) -> ChatResponse:
        """
        Main chat method
        
        Args:
            user_message: User's input
            session_id: Chat session ID (create if None)
            context_type: Type of context (jobs, cv, matching)
            use_rag: Override RAG setting
            
        Returns:
            ChatResponse with bot's reply
        """
        try:
            # Create session if needed
            if not session_id:
                session_id = self.create_session()
            
            # Add user message to history
            self.add_to_history(session_id, ChatRole.USER, user_message)
            
            logger.info(f"💬 Chat message: {user_message[:50]}... (session={session_id})")
            
            # Determine if RAG should be used
            use_rag_this_time = use_rag if use_rag is not None else self.enable_rag
            
            # Retrieve context (if RAG enabled)
            context = ""
            sources = []
            if use_rag_this_time:
                context, sources = self._retrieve_context(user_message)
            
            # Build system prompt
            system_prompt = self._build_system_prompt(context_type)
            
            # Build conversation context
            conv_context = self._build_conversation_context(session_id)
            
            # Build full message for LLM
            if context and use_rag_this_time:
                full_prompt = f"""{conv_context}

CONTEXT từ tài liệu:
{context}

Câu hỏi hiện tại:
{user_message}

Hãy trả lời dựa trên context trên. Nếu context không liên quan, hãy nói rõ.
"""
            else:
                full_prompt = f"""{conv_context}

{user_message}
"""
            
            # Generate response
            bot_response = self.llm_service.generate_response(
                full_prompt,
                system_prompt=system_prompt
            )
            
            # Add bot response to history
            self.add_to_history(
                session_id, 
                ChatRole.ASSISTANT, 
                bot_response,
                sources=sources if sources else None
            )
            
            # Create response
            chat_response = ChatResponse(
                response=bot_response,
                session_id=session_id,
                sources=sources,
                confidence_score=self._calculate_confidence(sources)
            )
            
            logger.info(f"✅ Response generated ({len(bot_response)} chars)")
            return chat_response
            
        except Exception as e:
            logger.error(f"❌ Chat error: {e}")
            raise
    
    def _calculate_confidence(self, sources: List[Dict[str, Any]]) -> Optional[float]:
        """Calculate confidence score based on sources"""
        if not sources:
            return None
        
        if not sources[0].get("similarity"):
            return None
        
        # Average similarity of top sources
        similarity_scores = [
            s.get("similarity", 0) 
            for s in sources[:3] 
            if s.get("similarity")
        ]
        
        if not similarity_scores:
            return None
        
        return sum(similarity_scores) / len(similarity_scores)
    
    def switch_collection(self, collection_name: str):
        """Switch to different collection"""
        self.collection_name = collection_name
        self.retrieval_service.change_collection(collection_name)
        logger.info(f"✅ Switched to collection: {collection_name}")


# Global chatbot instance
_chatbot: Optional[ChatbotRAG] = None


def get_chatbot(
    collection_name: str = "jobs",
    enable_rag: bool = True
) -> ChatbotRAG:
    """Get or create global chatbot instance"""
    global _chatbot
    if _chatbot is None:
        _chatbot = ChatbotRAG(
            collection_name=collection_name,
            enable_rag=enable_rag
        )
    return _chatbot


def reset_chatbot():
    """Reset chatbot (for testing)"""
    global _chatbot
    _chatbot = None
