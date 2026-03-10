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
        self.k_documents = k_documents if k_documents > 3 else 5
        self.enable_rag = enable_rag
        
        # Initialize services
        self.llm_service = get_llm_service()
        self.retrieval_service = get_retrieval_service(collection_name)
        
        # Chat history storage (in-memory, could use database)
        self.chat_sessions: Dict[str, List[ChatHistoryItem]] = {}
        
        logger.info(f" ChatbotRAG initialized (collection={collection_name})")
    
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
        from services.conversation_service import get_recent_history

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
