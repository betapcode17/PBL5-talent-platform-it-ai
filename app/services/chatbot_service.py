# app/services/chatbot_service.py
"""
Chatbot Service - Main chatbot logic combining RAG + LLM.
Handles chat sessions, message processing, and conversation flow.
"""

import logging
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

from .llm_service import get_llm_service
from .retrieval_service import get_retrieval_service
from app.models.chatbot import ChatResponse, ChatHistoryItem, ChatRole, RAGContext
from app.prompts.chatbot_system_prompt import CHAT_SYSTEM_PROMPTS

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
        self.k_documents = k_documents if k_documents > 3 else 5
        self.enable_rag = enable_rag
        
        # Initialize services
        self.llm_service = get_llm_service()
        self.retrieval_service = get_retrieval_service(collection_name)
        
        # Chat history storage (in-memory, could use database)
        self.chat_sessions: Dict[str, List[ChatHistoryItem]] = {}
        
        logger.info(f"[INFO] ChatbotRAG initialized (collection={collection_name})")
    
    def create_session(self) -> str:
        """Create new chat session"""
        session_id = str(uuid.uuid4())
        self.chat_sessions[session_id] = []
        logger.info(f" Created session: {session_id}")
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
            logger.info(f" Cleared history for session: {session_id}")
            return True
        return False
    def _retrieve_context(self, query: str) -> tuple[str, List[Dict[str, Any]]]:
        """
        Retrieve context from ChromaDB using hybrid search.
        Uses enriched metadata directly — no PostgreSQL enrichment needed.
        """
        if not self.enable_rag:
            return "", []

        try:
            context_parts = []
            sources = []

            # Extract keyword-based filters
            filters = self._extract_filters(query)

            # Hybrid search: semantic + metadata filters
            try:
                doc_results = self.retrieval_service.hybrid_search(
                    query,
                    filters=filters,
                    k=self.k_documents
                )
                for doc in doc_results:
                    m = doc.metadata
                    context_parts.append(
                        f"Viec lam: {m.get('job_title', 'N/A')}\n"
                        f"  Cong ty: {m.get('company', 'N/A')}\n"
                        f"  Dia diem: {m.get('location', 'N/A')}\n"
                        f"  Luong: {m.get('salary', 'Thuong luong')}\n"
                        f"  Kinh nghiem: {m.get('experience', 'N/A')}\n"
                        f"  Ky nang: {m.get('skills', 'N/A')}\n"
                        f"  Hinh thuc: {m.get('work_type', 'N/A')} - {m.get('job_type', 'N/A')}\n"
                        f"  Cap bac: {m.get('level', 'N/A')}\n"
                        f"  URL: {m.get('url', 'N/A')}\n"
                    )
                    sources.append({
                        "id": doc.id,
                        "title": m.get('job_title'),
                        "company": m.get('company'),
                        "location": m.get('location'),
                        "salary": m.get('salary'),
                        "url": m.get('url'),
                        "similarity": doc.distance
                    })
            except Exception as e:
                logger.warning(f"Hybrid search failed: {e}")

            # If no results, add market stats from ChromaDB
            if not sources:
                try:
                    stats = self.retrieval_service.get_collection_stats()
                    context_parts.append(
                        f"Thong ke thi truong:\n"
                        f"  Tong viec lam dang tuyen: {stats.get('total_jobs', 'N/A')}\n"
                        f"  So cong ty: {stats.get('total_companies', 'N/A')}\n"
                        f"  Top nganh: {', '.join(c['name'] for c in stats.get('top_categories', [])[:5])}\n"
                        f"  Top ky nang: {', '.join(s['name'] for s in stats.get('top_skills', [])[:5])}\n"
                    )
                except Exception:
                    pass

            context = "\n---\n".join(context_parts) if context_parts else "Khong tim thay cong viec phu hop."
            return context, sources

        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return "", []

    def _extract_filters(self, message: str) -> Dict[str, Any]:
        """Extract metadata filters from message using keyword matching (no LLM call)."""
        msg = message.lower()
        filters: Dict[str, Any] = {}

        # Work type
        if "remote" in msg:
            filters["work_type"] = "remote"
        elif any(kw in msg for kw in ["onsite", "on-site", "tai van phong", "tại văn phòng"]):
            filters["work_type"] = "at_office"
        elif "hybrid" in msg:
            filters["work_type"] = "hybrid"

        # Job type
        if any(kw in msg for kw in ["full-time", "fulltime", "toàn thời gian", "toan thoi gian"]):
            filters["job_type"] = "Full-time"
        elif any(kw in msg for kw in ["part-time", "parttime", "bán thời gian", "ban thoi gian"]):
            filters["job_type"] = "Part-time"
        elif any(kw in msg for kw in ["intern", "thực tập", "thuc tap", "internship"]):
            filters["job_type"] = "Internship"

        return filters

    _AGGREGATE_KEYWORDS = [
        "bao nhiêu", "bao nhieu", "có bao nhiêu", "co bao nhieu",
        "thống kê", "thong ke", "top ", "xếp hạng", "xep hang",
        "nhiều nhất", "nhieu nhat", "ít nhất", "it nhat",
        "đếm", "dem", "count", "phân bố", "phan bo",
        "tỷ lệ", "ty le", "tổng cộng", "tong cong",
        "cao nhất", "cao nhat", "thấp nhất", "thap nhat",
        "phần trăm", "phan tram",
    ]

    def _is_aggregate_query(self, message: str) -> bool:
        """Check if the message asks for counts / statistics / rankings."""
        msg = message.lower()
        return any(kw in msg for kw in self._AGGREGATE_KEYWORDS)

    def _handle_aggregate(self, message: str) -> tuple[str, List[Dict[str, Any]]]:
        """
        Handle aggregate/statistical queries using ChromaDB metadata.
        Returns (context_str, sources_list) for the LLM to format.
        """
        from collections import Counter

        stats = self.retrieval_service.get_collection_stats()
        total = stats.get("total_jobs", 0)
        context_parts = [f"Tong so viec lam trong he thong: {total}"]
        context_parts.append(f"So cong ty: {stats.get('total_companies', 0)}")
        sources: List[Dict[str, Any]] = [{"type": "aggregate", "total_jobs": total}]

        msg = message.lower()

        # Top skills
        if any(kw in msg for kw in ["kỹ năng", "ky nang", "skill"]):
            context_parts.append("Top ky nang duoc yeu cau nhieu nhat:")
            for s in stats.get("top_skills", [])[:10]:
                context_parts.append(f"  - {s['name']}: {s['count']} viec lam")

        # Top categories
        elif any(kw in msg for kw in ["ngành", "nganh", "category", "lĩnh vực", "linh vuc"]):
            context_parts.append("Phan bo theo nganh:")
            for c in stats.get("top_categories", [])[:10]:
                context_parts.append(f"  - {c['name']}: {c['count']} viec lam")

        # Top companies
        elif any(kw in msg for kw in ["công ty", "cong ty", "company"]):
            all_data = self.retrieval_service.aggregate_search()
            comp_counter: Counter = Counter(
                m.get("company", "") for m in all_data.get("metadatas", []) if m.get("company")
            )
            context_parts.append("Top cong ty co nhieu viec lam:")
            for comp, count in comp_counter.most_common(10):
                context_parts.append(f"  - {comp}: {count} viec lam")

        # Work type distribution
        elif any(kw in msg for kw in ["remote", "onsite", "hybrid", "hình thức", "hinh thuc"]):
            for wt in stats.get("work_type_dist", []):
                context_parts.append(f"  - {wt['name']}: {wt['count']} viec lam")

        # Location distribution
        elif any(kw in msg for kw in ["thành phố", "thanh pho", "địa điểm", "dia diem", "location"]):
            context_parts.append("Phan bo theo khu vuc:")
            for loc in stats.get("top_locations", [])[:10]:
                context_parts.append(f"  - {loc['name']}: {loc['count']} viec lam")

        else:
            # Generic overview
            context_parts.append("Top nganh: " + ", ".join(
                f"{c['name']}({c['count']})" for c in stats.get("top_categories", [])[:5]
            ))
            context_parts.append("Top ky nang: " + ", ".join(
                f"{s['name']}({s['count']})" for s in stats.get("top_skills", [])[:5]
            ))

        # Also add a few sample jobs from semantic search for context
        try:
            sample_docs = self.retrieval_service.hybrid_search(message, k=3)
            if sample_docs:
                context_parts.append("\nMot so viec lam lien quan:")
                for doc in sample_docs:
                    m = doc.metadata
                    context_parts.append(
                        f"  - {m.get('job_title', 'N/A')} | {m.get('company', 'N/A')} | "
                        f"{m.get('location', 'N/A')} | {m.get('salary', 'N/A')}"
                    )
        except Exception:
            pass

        return "\n".join(context_parts), sources

    # ----- Intent Detection -----

    _INTENT_KEYWORDS = {
        "cv": [
            "cv", "resume", "hồ sơ", "ho so",
            "phân tích cv", "phan tich cv",
            "cải thiện cv", "cai thien cv",
            "đánh giá cv", "danh gia cv",
            "viết cv", "viet cv",
            "sửa cv", "sua cv",
            "upload cv", "tải cv", "tai cv",
            "review cv", "xem cv",
            "nâng cấp cv", "nang cap cv",
        ],
        "matching": [
            "phù hợp", "phu hop",
            "match", "matching",
            "so khớp", "so khop",
            "ghép nối", "ghep noi",
            "có hợp không", "co hop khong",
            "nên ứng tuyển", "nen ung tuyen",
            "mức độ phù hợp", "muc do phu hop",
        ],
        "career": [
            "lộ trình", "lo trinh",
            "roadmap", "career path", "career",
            "nên học", "nen hoc",
            "học gì", "hoc gi",
            "chuyển ngành", "chuyen nganh",
            "định hướng", "dinh huong",
            "phỏng vấn", "phong van", "interview",
            "phát triển sự nghiệp", "phat trien su nghiep",
            "tư vấn nghề", "tu van nghe",
            "kỹ năng cần", "ky nang can",
            "cải thiện kỹ năng", "cai thien ky nang",
            "lộ trình học", "lo trinh hoc",
            "nên làm gì", "nen lam gi",
        ],
        "jobs": [
            "tìm việc", "tim viec",
            "việc làm", "viec lam",
            "tuyển dụng", "tuyen dung",
            "lương", "luong", "salary",
            "mức lương", "muc luong",
            "ứng tuyển", "ung tuyen",
            "thị trường", "thi truong",
            "tìm job", "tim job",
            "nhà tuyển dụng", "nha tuyen dung",
            "remote", "onsite", "hybrid",
            "fulltime", "full-time", "part-time", "parttime",
            "intern", "fresher", "junior", "senior",
            "cần tuyển", "can tuyen",
            "tìm công việc", "tim cong viec",
            # Merged from SQL intent — aggregate/statistical queries
            "bao nhiêu", "bao nhieu",
            "có bao nhiêu", "co bao nhieu",
            "thống kê", "thong ke",
            "top", "xếp hạng", "xep hang",
            "nhiều nhất", "nhieu nhat",
            "ít nhất", "it nhat",
            "trung bình", "trung binh",
            "tổng cộng", "tong cong",
            "đếm", "dem", "count",
            "so sánh", "so sanh",
            "phân bố", "phan bo",
            "tỷ lệ", "ty le",
            "cao nhất", "cao nhat",
            "thấp nhất", "thap nhat",
            "phần trăm", "phan tram",
            "dữ liệu", "du lieu",
        ],
    }

    def _detect_intent(self, message: str) -> str:
        """
        Auto-detect user intent from message.
        Priority: matching > cv > career > jobs > default
        Returns: jobs | cv | matching | career | default
        """
        msg = message.lower()

        # 1. Keyword-based (instant, free)
        for intent in ("matching", "cv", "career", "jobs"):
            if any(kw in msg for kw in self._INTENT_KEYWORDS[intent]):
                logger.info(f" Intent detected (keyword): {intent}")
                return intent

        # 2. LLM classification (fallback for ambiguous messages)
        try:
            classification = self._classify_with_llm(message)
            if classification in ("jobs", "cv", "matching", "career"):
                logger.info(f" Intent detected (LLM): {classification}")
                return classification
        except Exception as e:
            logger.warning(f"LLM intent classification failed: {e}")

        logger.info(" Intent detected: default")
        return "default"

    def _classify_with_llm(self, message: str) -> str:
        """Use LLM to classify ambiguous messages into intent categories."""
        prompt = (
            "Phan loai tin nhan sau vao MOT trong cac loai:\n"
            "- jobs: tim viec, hoi ve cong viec, goi y viec lam, thong ke viec lam, bao nhieu, top, xep hang\n"
            "- cv: hoi ve CV, cai thien ho so, phan tich CV\n"
            "- matching: so khop CV voi viec, do phu hop\n"
            "- career: lo trinh nghe nghiep, phat trien ky nang, dinh huong\n"
            "- default: chao hoi, cau hoi chung\n\n"
            f'Tin nhan: "{message}"\n\n'
            "Chi tra loi MOT tu: jobs, cv, matching, career, hoac default"
        )
        result = self.llm_service.generate_response(
            prompt,
            system_prompt="You are a classifier. Reply with exactly one word."
        )
        return result.strip().lower().split()[0]

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
            role = " Bạn" if msg.role == ChatRole.USER else " Chatbot"
            context_lines.append(f"{role}: {msg.content[:200]}")
        
        return "\n".join(context_lines)
    
    def chat(
        self,
        user_message: str,
        session_id: Optional[str] = None,
        context_type: str = "auto",
        use_rag: Optional[bool] = None
    ) -> ChatResponse:
        """
        Main chat method.
        When context_type="auto", intent is detected from the message.
        """
        try:
            # Create session if needed
            if not session_id:
                session_id = self.create_session()
            
            # Add user message to history
            self.add_to_history(session_id, ChatRole.USER, user_message)
            
            logger.info(f" Chat message: {user_message[:50]}... (session={session_id})")
            
            # --- Intent detection ---
            if context_type == "auto":
                detected_intent = self._detect_intent(user_message)
            else:
                detected_intent = context_type
            
            logger.info(f" Using intent: {detected_intent}")
            
            # Determine if RAG should be used
            use_rag_this_time = use_rag if use_rag is not None else self.enable_rag
            
            # --- Retrieve context based on intent ---
            context = ""
            sources = []
            if detected_intent == "jobs" and use_rag_this_time:
                if self._is_aggregate_query(user_message):
                    context, sources = self._handle_aggregate(user_message)
                else:
                    context, sources = self._retrieve_context(user_message)
            elif detected_intent == "career" and use_rag_this_time:
                # Career advice benefits from market stats (from ChromaDB)
                try:
                    stats = self.retrieval_service.get_collection_stats()
                    top_skills = ', '.join(s['name'] for s in stats.get('top_skills', [])[:10])
                    top_cats = ', '.join(c['name'] for c in stats.get('top_categories', [])[:5])
                    context = (
                        f"Thong ke thi truong hien tai:\n"
                        f"  Tong viec lam: {stats.get('total_jobs', 'N/A')}\n"
                        f"  So cong ty: {stats.get('total_companies', 'N/A')}\n"
                        f"  Top nganh: {top_cats}\n"
                        f"  Top ky nang: {top_skills}\n"
                    )
                except Exception:
                    pass
            
            # Build conversation context
            conv_context = self._build_conversation_context(session_id)

            # Build system prompt
            system_prompt = self._build_system_prompt(detected_intent)
            
            # --- Build full prompt based on intent ---
            if detected_intent == "jobs" and context:
                full_prompt = f"""{conv_context}

=== DU LIEU THUC TE TU HE THONG ===
{context}
=== HET DU LIEU ===

Cau hoi cua nguoi dung: {user_message}

Hay tra loi dua tren du lieu thuc te o tren. Chi dua tren du lieu thuc te, KHONG tu nghi ra.
"""
            elif detected_intent == "career" and context:
                full_prompt = f"""{conv_context}

=== THONG KE THI TRUONG ===
{context}
=== HET THONG KE ===

Cau hoi cua nguoi dung: {user_message}

Hay tu van dua tren thong ke thi truong thuc te o tren.
"""
            else:
                full_prompt = f"""{conv_context}

Cau hoi cua nguoi dung: {user_message}
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
                detected_intent=detected_intent,
                confidence_score=self._calculate_confidence(sources)
            )
            
            logger.info(f" Response generated ({len(bot_response)} chars)")
            return chat_response
            
        except Exception as e:
            logger.error(f" Chat error: {e}")
            raise
    
    def chat_with_conversation(
        self,
        user_message: str,
        conversation_id: str,
    ) -> Dict[str, Any]:
        """
        Process a user message within a persistent conversation.
        Returns dict with 'bot_response', 'sources', 'detected_intent'.
        Conversation history is read from the database.
        """
        from app.services.conversation_service import get_recent_history

        # Intent detection
        detected_intent = self._detect_intent(user_message)
        logger.info(f" Conversation {conversation_id} intent: {detected_intent}")

        # Retrieve context
        context = ""
        sources: List[Dict[str, Any]] = []
        if detected_intent == "jobs" and self.enable_rag:
            if self._is_aggregate_query(user_message):
                context, sources = self._handle_aggregate(user_message)
            else:
                context, sources = self._retrieve_context(user_message)
        elif detected_intent == "career" and self.enable_rag:
            try:
                stats = self.retrieval_service.get_collection_stats()
                top_skills = ', '.join(s['name'] for s in stats.get('top_skills', [])[:10])
                top_cats = ', '.join(c['name'] for c in stats.get('top_categories', [])[:5])
                context = (
                    f"Thong ke thi truong hien tai:\n"
                    f"  Tong viec lam: {stats.get('total_jobs', 'N/A')}\n"
                    f"  So cong ty: {stats.get('total_companies', 'N/A')}\n"
                    f"  Top nganh: {top_cats}\n"
                    f"  Top ky nang: {top_skills}\n"
                )
            except Exception:
                pass

        # Build conversation context from DB
        recent = get_recent_history(conversation_id, max_turns=5)
        conv_lines = []
        if recent:
            conv_lines.append("Cuộc trò chuyện trước đây:")
            for m in recent:
                role_label = " Bạn" if m["role"] == "user" else " Chatbot"
                conv_lines.append(f"{role_label}: {m['content'][:200]}")
        conv_context = "\n".join(conv_lines)

        # Build system prompt
        system_prompt = self._build_system_prompt(detected_intent)

        # Build full prompt
        if detected_intent == "jobs" and context:
            full_prompt = f"""{conv_context}

=== DU LIEU THUC TE TU HE THONG ===
{context}
=== HET DU LIEU ===

Cau hoi cua nguoi dung: {user_message}

Hay tra loi dua tren du lieu thuc te o tren. Chi dua tren du lieu thuc te, KHONG tu nghi ra.
"""
        elif detected_intent == "career" and context:
            full_prompt = f"""{conv_context}

=== THONG KE THI TRUONG ===
{context}
=== HET THONG KE ===

Cau hoi cua nguoi dung: {user_message}

Hay tu van dua tren thong ke thi truong thuc te o tren.
"""
        else:
            full_prompt = f"""{conv_context}

Cau hoi cua nguoi dung: {user_message}
"""

        # Generate LLM response
        bot_response = self.llm_service.generate_response(
            full_prompt,
            system_prompt=system_prompt
        )

        logger.info(f" Conversation response generated ({len(bot_response)} chars)")
        return {
            "bot_response": bot_response,
            "sources": sources,
            "detected_intent": detected_intent,
        }

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
        logger.info(f" Switched to collection: {collection_name}")


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


# Tool-aware chatbot wrapper
_tool_aware_chatbot: Optional['ToolAwareChatbotWrapper'] = None


class ToolAwareChatbotWrapper:
    """Wrapper that adds tool calling capabilities to the chatbot"""
    
    def __init__(self, chatbot: ChatbotRAG):
        self.chatbot = chatbot
        # Import and initialize tool-aware chatbot with backend API
        from app.services.tool_aware_chatbot import ToolAwareChatbot
        self.tool_aware_chatbot = ToolAwareChatbot(chatbot)
        self.debug_mode = True  # Enable decision tracing
    
    @property
    def jobs_service(self):
        """Proxy to backend API client"""
        return self.tool_aware_chatbot.backend_client
    
    async def search_jobs(self, **kwargs):
        """Proxy to backend API client search_jobs"""
        return await self.tool_aware_chatbot.backend_client.search_jobs(
            query=kwargs.get("query"),
            category=kwargs.get("category"),
            location=kwargs.get("location"),
            salary_min=kwargs.get("salary_min"),
            page=kwargs.get("page", 1),
            limit=kwargs.get("limit", 20),
        )
    
    def _count_tokens(self, text: str) -> int:
        """Rough estimate: ~4 chars = 1 token (for LLM counting)"""
        return max(1, len(text) // 4)
    
    def _log_decision_trace(self, stage: str, data: Dict[str, Any], result: Any = None):
        """Log decision-making process with structured format"""
        if not self.debug_mode:
            return
        
        trace_msg = f"[DECISION_TRACE] Stage: {stage} | Data: {data}"
        if result is not None:
            trace_msg += f" | Result: {result}"
        logger.info(trace_msg)
    
    def _classify_job_intent_with_llm(self, message: str) -> bool:
        """
        Use LLM to classify if message is asking about jobs.
        Called only if keyword matching fails (fallback).
        """
        try:
            prompt = f"""Người dùng có đang hỏi về việc làm không?

Câu hỏi: "{message}"

Trả lời CHỈ một từ: "có" hoặc "không"

Ví dụ:
- "Tìm Backend jobs ở Hà Nội" → có
- "Tôi muốn làm việc ở vị trí backend" → có
- "Có cơ hội nào cần tuyển dụng không?" → có
- "Kể về công ty của bạn" → không
- "Làm sao để viết CV tốt?" → không
"""
            
            result = self.chatbot.llm_service.generate_response(
                prompt,
                system_prompt="You are a classifier. Respond with exactly one word in Vietnamese."
            )
            
            # Check if response contains affirmative answer
            return "có" in result.lower() or "yes" in result.lower()
        
        except Exception as e:
            logger.warning(f"LLM intent classification failed: {e}. Defaulting to False.")
            return False
    
    def _extract_job_id(self, message: str) -> Optional[int]:
        """Extract job ID from message like 'Chi tiết công việc có ID: 6283' or 'Job 6283'"""
        import re
        # Patterns to match job IDs
        patterns = [
            r'id\s*[:=]\s*(\d+)',          # id: 6283 or id=6283
            r'id\s+(\d+)',                  # id 6283
            r'job\s*[:=]\s*(\d+)',          # job: 6283 or job=6283
            r'job\s+(\d+)',                 # job 6283
            r'công việc\s*:\s*(\d+)',       # công việc: 6283
            r'cong viec\s*:\s*(\d+)',       # cong viec: 6283
            r'#(\d+)',                      # #6283
        ]
        
        message_lower = message.lower()
        for pattern in patterns:
            match = re.search(pattern, message_lower)
            if match:
                return int(match.group(1))
        return None
    
    def _detect_job_detail_intent(self, message: str) -> Optional[Dict[str, Any]]:
        """Detect if user is asking for job details (not search)"""
        import re
        message_lower = message.lower()
        
        # Keywords that indicate detail/info request
        detail_keywords = [
            "chi tiết", "chi tiet",
            "thông tin", "thong tin",
            "thế nào", "the nao",
            "như thế nào", "nhu the nao",
            "giới thiệu", "gioi thieu",
            "mô tả", "mo ta",
            "yêu cầu", "yeu cau",
            "requirements",
        ]
        
        has_detail_keyword = any(kw in message_lower for kw in detail_keywords)
        
        if not has_detail_keyword:
            return None
        
        # Extract job ID
        job_id = self._extract_job_id(message)
        if not job_id:
            return None
        
        self._log_decision_trace(
            stage="intent_detection_detail",
            data={"message_length": len(message), "job_id": job_id},
            result="job_detail_intent"
        )
        
        return {"tool": "get_job_detail", "params": {"job_id": job_id}}
    
    def _detect_job_intent(self, message: str) -> Optional[Dict[str, Any]]:
        """Detect if user is asking about jobs"""
        import re
        message_lower = message.lower()
        
        # Keywords that indicate job search intent
        job_keywords = [
            "tìm công việc", "find job", "search job", "liệt kê job",
            "list job", "công việc", "job", "vị trí", "position",
            "tuyển", "hiring", "việc làm",
            # Technology roles and keywords
            "backend", "frontend", "fullstack", "devops", "qa",
            "data", "mobile", "python", "java", "javascript",
            "react", "node", "spring", "docker", "kubernetes",
            # English variations
            "search", "find", "developer", "engineer", "intern", "recruit"
        ]
        
        has_job_keyword = any(kw in message_lower for kw in job_keywords)
        
        # Log intent detection
        self._log_decision_trace(
            stage="intent_detection",
            data={"message_length": len(message), "has_job_keyword": has_job_keyword},
            result="job_search" if has_job_keyword else "no_job_keyword"
        )
        
        # ✅ ADD: LLM fallback for ambiguous queries
        if not has_job_keyword:
            logger.info(" Keyword match failed, trying LLM classification...")
            
            is_job_related = self._classify_job_intent_with_llm(message)
            
            self._log_decision_trace(
                stage="intent_detection_llm_fallback",
                data={"message_length": len(message), "llm_result": is_job_related},
                result="job_search_llm" if is_job_related else "no_job_keyword_llm"
            )
            
            if not is_job_related:
                return None
            
            logger.info(" LLM confirmed: This is a job-related query")
        
        # Extract parameters
        params = {}
        
        # Location extraction
        location_map = {
            "hà nội": "Hanoi", "hanoi": "Hanoi",
            "sài gòn": "HCM", "hcm": "HCM", "tp hcm": "HCM",
            "đà nẵng": "Da Nang", "da nang": "Da Nang",
        }
        for vn, en in location_map.items():
            if vn in message_lower:
                params["location"] = en
                break
        
        # Category extraction
        category_map = {
            "backend": "Backend",
            "frontend": "Frontend",
            "fullstack": "Fullstack",
            "devops": "DevOps",
            "qa": "QA",
            "data": "Data",
        }
        for keyword, category in category_map.items():
            if keyword in message_lower:
                params["category"] = category
                break
        
        # ✅ IMPROVED: Better salary extraction
        salary_patterns = [
            r'(\d+)\s*(?:triệu|tr)',              # 50 triệu, 50tr
            r'(\d+)\s*[mk]\b',                    # 50m, 50k
            r'(\d+)\s*[,-]\s*(\d+)\s*[mk]',       # 50-60m
        ]
        for pattern in salary_patterns:
            salary_match = re.search(pattern, message_lower)
            if salary_match:
                if salary_match.lastindex == 1:
                    amount = salary_match.group(1)
                    params["salary_min"] = f"{amount}m"
                else:
                    # Range: use minimum
                    amount = salary_match.group(1)
                    params["salary_min"] = f"{amount}m"
                break
        
        # ✅ NEW: Experience level extraction
        level_map = {
            "fresher|mới ra trường": "Fresher",
            "junior": "Junior", 
            "senior": "Senior",
            "lead": "Lead",
            "manager": "Manager",
        }
        for keywords, level in level_map.items():
            if any(kw in message_lower for kw in keywords.split('|')):
                params["level"] = level
                break
        
        # Extract search query - IMPROVED STRATEGY:
        # If category is already extracted, use empty query (backend will filter by category)
        # Otherwise, use the full message or cleaned message
        if "category" in params:
            # Category filter exists, use empty query for backend to return all jobs in that category
            params["query"] = ""
        else:
            # No category, need to build meaningful query from message
            # Remove common filler words but preserve meaningful keywords
            query = message_lower
            for location in location_map.keys():
                query = query.replace(location, "")
            
            # Remove common search words to avoid empty queries
            filler_words = ["tìm", "tim", "việc", "viec", "công", "cong", "jobs", "job", "làm"]
            for word in filler_words:
                query = query.replace(word, "")
            
            # If query is now empty, use the original message (backend will search on full text)
            query_cleaned = " ".join(query.split()).strip()
            params["query"] = query_cleaned or message_lower
        
        # Log extracted parameters
        self._log_decision_trace(
            stage="parameter_extraction",
            data=params,
            result="search_jobs_ready"
        )
        
        return {"tool": "search_jobs", "params": params}
    
    def _format_job_detail(self, job: Dict[str, Any]) -> str:
        """Format detailed job information for chat response"""
        response = f"📋 **{job.get('title', 'N/A')}** (ID: {job.get('id', 'N/A')})\n\n"
        
        # Company info
        company = job.get("company", {})
        if isinstance(company, dict):
            response += f"🏢 Công ty: {company.get('company_name', 'N/A')}\n"
            if company.get("company_website_url"):
                response += f"   Website: {company.get('company_website_url')}\n"
            if company.get("city"):
                response += f"   Địa điểm: {company.get('city', 'N/A')}, {company.get('country', 'N/A')}\n"
        
        response += "\n"
        
        # Salary
        salary = job.get("salary", "N/A")
        salary_range = job.get("salaryRange", {})
        if salary_range.get("min") and salary_range.get("max"):
            response += f"💰 Lương: {salary_range['min']} - {salary_range['max']}\n"
        elif salary and salary != "N/A":
            response += f"💰 Lương: {salary}\n"
        
        # Category & Job Type
        category = job.get("category", {})
        job_type = job.get("jobType", {})
        if isinstance(category, dict):
            response += f"📌 Danh mục: {category.get('name', 'N/A')}\n"
        if isinstance(job_type, dict):
            response += f"⏰ Loại việc: {job_type.get('job_type', 'N/A')}\n"
        
        # Description
        if job.get("description"):
            response += f"\n📝 **Mô tả công việc:**\n{job['description']}\n"
        
        # Requirements
        requirements = job.get("requirements", [])
        if requirements:
            response += f"\n✅ **Yêu cầu:**\n"
            for req in requirements[:10]:  # Limit to 10 requirements
                if req.strip():
                    response += f"  • {req}\n"
            if len(requirements) > 10:
                response += f"  ... và {len(requirements) - 10} yêu cầu khác\n"
        
        # Additional info
        if job.get("experience"):
            response += f"\n👤 Kinh nghiệm: {job['experience']}\n"
        if job.get("level"):
            response += f"📊 Level: {job['level']}\n"
        if job.get("education"):
            response += f"🎓 Học vấn: {job['education']}\n"
        if job.get("numberOfHires"):
            response += f"👥 Số lượng tuyển: {job['numberOfHires']}\n"
        if job.get("deadline"):
            response += f"📅 Hạn chót: {job['deadline']}\n"
        
        return response
    
    def _format_jobs_result(self, jobs: List[Dict], total: int) -> str:
        """Format job results for chat response"""
        if not jobs:
            return "Không tìm thấy công việc phù hợp. Vui lòng thử từ khóa khác."
        
        response = f"Tìm thấy {total} công việc:\n\n"
        
        for i, job in enumerate(jobs, 1):
            title = job.get("title", "N/A")
            # Handle nested company object from backend
            company_name = job.get("company", {}).get("company_name", "N/A") if isinstance(job.get("company"), dict) else job.get("company", "N/A")
            salary = job.get("salary", "Thương lượng")
            location = job.get("location", "N/A")
            job_id = job.get("id", "N/A")
            
            response += f"\n{i}. {title} (ID: {job_id})"
            response += f"\n   Công ty: {company_name}"
            response += f"\n   Lương: {salary}" if salary != "N/A" else ""
            response += f"\n   Địa điểm: {location}\n"
        
        response += "\n\nGợi ý: Bạn có thể yêu cầu chi tiết hơn bằng cách nói ID công việc hoặc hỏi về yêu cầu công việc."
        
        return response
    
    async def chat_with_tools(self, user_message: str, conversation_id: str, file_ids: Optional[List[str]] = None) -> Dict[str, Any]: # type: ignore
        """Chat with tool integration and debug decision tracing"""
        try:
            # ===== STAGE 1: Token Counting =====
            input_tokens = self._count_tokens(user_message)
            self._log_decision_trace(
                stage="token_counting",
                data={"input_message_length": len(user_message), "estimated_input_tokens": input_tokens},
                result="tokens_counted"
            )
            
            # ===== STAGE 2: Intent Detection =====
            # First, check if user is asking for job details (by ID)
            job_detail_intent = self._detect_job_detail_intent(user_message)
            
            if job_detail_intent and job_detail_intent["tool"] == "get_job_detail":
                # ===== STAGE 3A: Get Job Detail =====
                job_id = job_detail_intent["params"]["job_id"]
                
                self._log_decision_trace(
                    stage="tool_decision",
                    data={"detected_tool": "get_job_detail", "job_id": job_id},
                    result="tool_call_approved"
                )
                
                try:
                    # Call backend API to get job details
                    job_detail = await self.jobs_service.get_job_details(job_id) # type: ignore
                    
                    # Check if there's an error
                    if "error" in job_detail:
                        error_response = f"Xin lỗi, không tìm thấy công việc với ID {job_id}."
                        output_tokens = self._count_tokens(error_response)
                        return {
                            "bot_response": error_response,
                            "sources": [],
                            "detected_intent": "job_detail_error",
                            "tool_used": "get_job_detail",
                            "debug_info": {
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens,
                                "total_tokens": input_tokens + output_tokens,
                                "error": job_detail.get("error", "Unknown error")
                            }
                        }
                    
                    # ===== STAGE 4A: Format Result =====
                    formatted_response = self._format_job_detail(job_detail)
                    
                    # ===== STAGE 5A: Token Output Counting =====
                    output_tokens = self._count_tokens(formatted_response)
                    total_tokens = input_tokens + output_tokens
                    
                    self._log_decision_trace(
                        stage="response_generation",
                        data={"output_length": len(formatted_response), "estimated_output_tokens": output_tokens},
                        result="response_ready"
                    )
                    
                    # ===== STAGE 6A: Stop Reason =====
                    self._log_decision_trace(
                        stage="stop_reason",
                        data={"job_detail_found": True},
                        result="stop_reason_job_detail_success"
                    )
                    
                    return {
                        "bot_response": formatted_response,
                        "sources": [],
                        "detected_intent": "job_detail",
                        "tool_used": "get_job_detail",
                        "debug_info": {
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": total_tokens,
                            "job_id": job_id
                        }
                    }
                except Exception as e:
                    logger.error(f"Error getting job detail for ID {job_id}: {e}")
                    error_response = f"Xin lỗi, không tìm thấy công việc với ID {job_id}. Vui lòng kiểm tra lại."
                    output_tokens = self._count_tokens(error_response)
                    return {
                        "bot_response": error_response,
                        "sources": [],
                        "detected_intent": "job_detail_error",
                        "tool_used": "get_job_detail",
                        "debug_info": {
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": input_tokens + output_tokens,
                            "error": str(e)
                        }
                    }
            
            # Otherwise, check for job search intent
            job_intent = self._detect_job_intent(user_message)
            
            if job_intent and job_intent["tool"] == "search_jobs":
                # ===== STAGE 3: Tool Call =====
                self._log_decision_trace(
                    stage="tool_decision",
                    data={"detected_tool": "search_jobs", "params": job_intent["params"]},
                    result="tool_call_approved"
                )
                
                # Call jobs service to search
                search_params = job_intent["params"]
                jobs_result = await self.jobs_service.search_jobs(
                    query=search_params.get("query", ""),
                    limit=search_params.get("limit", 100),
                    category=search_params.get("category"),
                    location=search_params.get("location"),
                    salary_min=search_params.get("salary_min")
                )
                
                # ===== STAGE 4: Format Results =====
                # Backend returns: {"jobs": [...], "total": N, "page": N, "limit": N}
                jobs_data = jobs_result.get("jobs", []) if isinstance(jobs_result, dict) else []
                total_jobs = jobs_result.get("total", len(jobs_data)) if isinstance(jobs_result, dict) else 0
                
                formatted_response = self._format_jobs_result(jobs_data, total=total_jobs)
                
                # ===== STAGE 5: Token Output Counting =====
                output_tokens = self._count_tokens(formatted_response)
                total_tokens = input_tokens + output_tokens
                
                self._log_decision_trace(
                    stage="response_generation",
                    data={"output_length": len(formatted_response), "estimated_output_tokens": output_tokens, "total_tokens": total_tokens},
                    result="response_ready"
                )
                
                # ===== STAGE 6: Stop Reason =====
                self._log_decision_trace(
                    stage="stop_reason",
                    data={"jobs_found": len(jobs_data), "search_successful": len(jobs_data) > 0},
                    result="stop_reason_tool_success"
                )
                
                return {
                    "bot_response": formatted_response,
                    "sources": [],
                    "detected_intent": "job_search",
                    "tool_used": "search_jobs",
                    "debug_info": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens,
                        "jobs_count": len(jobs_data)
                    }
                }
            
            else:
                # ===== STAGE 3 (ALT): No Tool, Use RAG =====
                self._log_decision_trace(
                    stage="tool_decision",
                    data={"job_intent_detected": False},
                    result="fallback_to_rag"
                )
                
                # Use regular chatbot
                logger.info("No job intent detected, using RAG chatbot")
                rag_result = self.chatbot.chat_with_conversation(
                    user_message=user_message,
                    conversation_id=conversation_id
                )
                
                # Add token counting to RAG result
                output_tokens = self._count_tokens(rag_result.get("bot_response", ""))
                total_tokens = input_tokens + output_tokens
                
                self._log_decision_trace(
                    stage="stop_reason",
                    data={"rag_fallback": True},
                    result="stop_reason_rag_used"
                )
                
                rag_result["debug_info"] = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "used_rag": True
                }
                
                return rag_result
        
        except Exception as e:
            logger.error(f"Error in tool-aware chat: {e}")
            
            # ===== ERROR STAGE =====
            self._log_decision_trace(
                stage="error_handling",
                data={"exception": str(e)},
                result="fallback_triggered"
            )
            
            # Fallback to regular chatbot
            try:
                return self.chatbot.chat_with_conversation(
                    user_message=user_message,
                    conversation_id=conversation_id
                )
            except Exception as fallback_error:
                logger.error(f"Fallback chatbot error: {fallback_error}")
                return {
                    "bot_response": f"Xin lỗi, đã xảy ra lỗi: {str(e)}",
                    "sources": [],
                    "detected_intent": "error",
                    "debug_info": {"error": str(e)}
                }
    
    async def chat_with_rag_only(self, user_message: str, conversation_id: str) -> Dict[str, Any]:
        """
        Chat using ONLY RAG (no tool calling) for job queries.
        This is used to benchmark performance vs tool_calling.
        
        Features:
        - Uses ChromaDB vector search for job queries instead of API calls
        - All jobs pre-indexed, instant retrieval
        - Compare speed: RAG vs Tool Calling
        """
        import time
        
        try:
            # ===== STAGE 1: Token Counting =====
            input_tokens = self._count_tokens(user_message)
            start_time = time.time()
            
            self._log_decision_trace(
                stage="rag_only_start",
                data={"input_message_length": len(user_message)},
                result="rag_retrieval_mode"
            )
            
            # ===== STAGE 2: Intent Detection (Fast, no LLM) =====
            detected_intent = self.chatbot._detect_intent(user_message)
            
            logger.info(f"[RAG] Intent detected: {detected_intent}")
            
            # ===== STAGE 3: RAG Retrieval (for jobs intent) =====
            rag_start = time.time()
            
            context = ""
            sources = []
            if detected_intent == "jobs" and self.chatbot.enable_rag:
                # Check if it's aggregate query
                if self.chatbot._is_aggregate_query(user_message):
                    logger.info("[RAG] Aggregate query detected")
                    context, sources = self.chatbot._handle_aggregate(user_message)
                else:
                    logger.info("[RAG] Regular job search via vector similarity")
                    context, sources = self.chatbot._retrieve_context(user_message)
            elif detected_intent == "career" and self.chatbot.enable_rag:
                try:
                    stats = self.chatbot.retrieval_service.get_collection_stats()
                    top_skills = ', '.join(s['name'] for s in stats.get('top_skills', [])[:10])
                    top_cats = ', '.join(c['name'] for c in stats.get('top_categories', [])[:5])
                    context = (
                        f"Thong ke thi truong hien tai:\n"
                        f"  Tong viec lam: {stats.get('total_jobs', 'N/A')}\n"
                        f"  So cong ty: {stats.get('total_companies', 'N/A')}\n"
                        f"  Top nganh: {top_cats}\n"
                        f"  Top ky nang: {top_skills}\n"
                    )
                except Exception as e:
                    logger.warning(f"[RAG] Career stats retrieval failed: {e}")
            
            rag_time = time.time() - rag_start
            
            self._log_decision_trace(
                stage="rag_retrieval",
                data={"rag_time_ms": rag_time * 1000, "sources_count": len(sources)},
                result="rag_retrieval_complete"
            )
            
            # ===== STAGE 4: LLM Response Generation =====
            llm_start = time.time()
            
            # Build conversation context from DB
            from app.services.conversation_service import get_recent_history
            recent = get_recent_history(conversation_id, max_turns=5)
            conv_lines = []
            if recent:
                conv_lines.append("Cuộc trò chuyện trước đây:")
                for m in recent:
                    role_label = " Bạn" if m["role"] == "user" else " Chatbot"
                    conv_lines.append(f"{role_label}: {m['content'][:200]}")
            conv_context = "\n".join(conv_lines)
            
            # Build system prompt
            system_prompt = self.chatbot._build_system_prompt(detected_intent)
            
            # Build full prompt with RAG context
            if detected_intent == "jobs" and context:
                full_prompt = f"""{conv_context}

=== DU LIEU TIM KIEM CHROMA (RAG) ===
{context}
=== HET DU LIEU ===

Cau hoi cua nguoi dung: {user_message}

Hay tra loi dua tren du lieu tim kiem o tren. Chi dua tren du lieu thuc te, KHONG tu nghi ra.
"""
            elif detected_intent == "career" and context:
                full_prompt = f"""{conv_context}

=== THONG KE THI TRUONG (TU CHROMA) ===
{context}
=== HET THONG KE ===

Cau hoi cua nguoi dung: {user_message}

Hay tu van dua tren thong ke thi truong thuc te o tren.
"""
            else:
                full_prompt = f"""{conv_context}

Cau hoi cua nguoi dung: {user_message}
"""
            
            # Generate LLM response
            bot_response = self.chatbot.llm_service.generate_response(
                full_prompt,
                system_prompt=system_prompt
            )
            
            llm_time = time.time() - llm_start
            total_time = time.time() - start_time
            
            # ===== STAGE 5: Token Output Counting =====
            output_tokens = self._count_tokens(bot_response)
            total_tokens = input_tokens + output_tokens
            
            self._log_decision_trace(
                stage="response_complete",
                data={
                    "output_length": len(bot_response),
                    "llm_time_ms": llm_time * 1000,
                    "rag_time_ms": rag_time * 1000,
                    "total_time_ms": total_time * 1000,
                    "estimated_output_tokens": output_tokens
                },
                result="response_ready"
            )
            
            # Return with detailed performance metrics
            return {
                "bot_response": bot_response,
                "sources": sources,
                "detected_intent": detected_intent,
                "method": "rag_only",
                "debug_info": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "performance_metrics": {
                        "rag_retrieval_ms": round(rag_time * 1000, 2),
                        "llm_generation_ms": round(llm_time * 1000, 2),
                        "total_latency_ms": round(total_time * 1000, 2),
                        "sources_count": len(sources),
                    }
                }
            }
        
        except Exception as e:
            logger.error(f"[RAG] Error in RAG-only chat: {e}")
            
            # Fallback to regular chatbot
            try:
                return self.chatbot.chat_with_conversation(
                    user_message=user_message,
                    conversation_id=conversation_id
                )
            except Exception as fallback_error:
                logger.error(f"[RAG] Fallback error: {fallback_error}")
                return {
                    "bot_response": f"Xin lỗi, đã xảy ra lỗi: {str(e)}",
                    "sources": [],
                    "detected_intent": "error",
                    "method": "rag_only_error",
                    "debug_info": {"error": str(e)}
                }
    
    async def chat_with_tools(self, user_message: str, conversation_id: str, file_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Delegate to ToolAwareChatbot.chat_with_tools for CV analysis and tool handling"""
        logger.info(f"[WRAPPER] chat_with_tools called with file_ids: {file_ids}")
        # Delegate to the underlying ToolAwareChatbot instance which has CV analysis logic
        return await self.tool_aware_chatbot.chat_with_tools(
            user_message=user_message,
            conversation_id=conversation_id,
            file_ids=file_ids
        )


def get_tool_aware_chatbot() -> ToolAwareChatbotWrapper:
    """Get tool-aware chatbot wrapper"""
    global _tool_aware_chatbot
    if _tool_aware_chatbot is None:
        base_chatbot = get_chatbot()
        _tool_aware_chatbot = ToolAwareChatbotWrapper(base_chatbot)
    return _tool_aware_chatbot
