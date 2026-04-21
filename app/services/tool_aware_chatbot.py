"""
Enhanced Chatbot with Tool Integration (v5)
- Query normalization & standardization
- Improved prompt for better parsing
- Retry logic for failed requests
- Better async error handling
- Connection pooling
- Timeout protection
- Improved fallback strategy
"""

import logging
import json
import re
import asyncio
from typing import Optional, Dict, Any, List
import httpx # type: ignore

from app.services.backend_api_client import get_backend_client
from app.services.llm_service import get_llm_service
from app.prompts.query_normalizer import QueryNormalizer, normalize_query
from app.prompts.improved_job_parser_prompt import (
    get_improved_job_parser_prompt,
    get_parsing_instructions,
    get_fallback_rules
)
from app.prompts.job_intent_parser_prompt import (
    get_job_keywords,
    get_job_categories,
    get_job_locations
)

logger = logging.getLogger(__name__)


class ToolAwareChatbot:
    """Chatbot with retry logic and robust error handling"""
    
    def __init__(self, chatbot, jobs_tools=None, http_client: httpx.AsyncClient = None, headers: dict = None): # type: ignore
        """Initialize with retry configuration"""
        self.chatbot = chatbot
        self.backend_client = get_backend_client()
        self.llm_service = get_llm_service()
        self.max_retries = 2
        self.retry_delay = 0.3  # seconds
        self.query_normalizer = QueryNormalizer()
        
        logger.info("[OK] ToolAwareChatbot v5 initialized with query normalization & improved parser")
    
    async def _parse_message_with_ollama_timeout(self, message: str, timeout_sec: float = 3.0) -> Dict[str, Any]:
        """Parse message with query normalization + improved prompt + timeout + fallback"""
        try:
            # Step 1: Normalize query
            normalized_result = self.query_normalizer.normalize(message)
            normalized_query = normalized_result["standardized"]
            
            logger.info(f"[NORMALIZE] Original: {message[:50]}")
            logger.info(f"[NORMALIZE] Standardized: {normalized_query[:80]}")
            
            # Step 2: Use improved prompt
            improved_prompt = get_improved_job_parser_prompt()
            full_prompt = f"""{improved_prompt}

User Query (Original): "{message}"
User Query (Normalized): "{normalized_query}"
Extracted Info: {json.dumps(normalized_result['extracted'], ensure_ascii=False)}

Now parse and return JSON:"""
            
            logger.info(f"[CALL] Ollama parse (timeout={timeout_sec}s)")
            
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.llm_service.generate_response,
                        full_prompt,
                        None,
                        0.1
                    ),
                    timeout=timeout_sec
                )
            except asyncio.TimeoutError:
                logger.warning(f"[WARNING] Ollama timeout, using regex with normalized data")
                return self._parse_message_with_improved_regex(message, normalized_result)
            
            # Step 3: Extract JSON
            response_clean = response.strip()
            if "```json" in response_clean:
                response_clean = response_clean.split("```json")[1].split("```")[0].strip()
            elif "```" in response_clean:
                response_clean = response_clean.split("```")[1].split("```")[0].strip()
            
            parsed = json.loads(response_clean)
            logger.info(f"[OK] Ollama parsed with improved prompt")
            return parsed
        
        except (json.JSONDecodeError, IndexError, ValueError) as e:
            logger.warning(f"[WARNING] JSON parse failed: {str(e)[:50]}, using improved regex")
            return self._parse_message_with_improved_regex(message, normalized_result)
        except Exception as e:
            logger.warning(f"[WARNING] Ollama error: {str(e)[:50]}, using improved regex")
            return self._parse_message_with_improved_regex(message, normalized_result)
    
    def _parse_message_with_improved_regex(self, message: str, normalized_result: Optional[Dict] = None) -> Dict[str, Any]:
        """Improved regex parsing using normalized data"""
        if normalized_result is None:
            normalized_result = self.query_normalizer.normalize(message)
        
        extracted = normalized_result["extracted"]
        message_lower = message.lower()
        
        # Build result from normalized data
        result = {
            "has_job_query": False,
            "tool": None,
            "category": extracted.get("category"),
            "location": extracted.get("location"),
            "keywords": ", ".join(extracted.get("keywords", [])) if extracted.get("keywords") else None,
            "salary_min": extracted.get("salary_min"),
            "salary_max": None,
            "level": extracted.get("level"),
            "employment_type": extracted.get("employment_type"),
            "job_id": extracted.get("job_id"),
            "company_name": extracted.get("company_name"),
            "limit": 20,
            "page": 1,
            "confidence": 0.7,  # Regex parsing = lower confidence
            "normalized_query": normalized_result["standardized"],
        }
        
        # Check for job query keywords
        job_keywords = get_job_keywords()
        if not any(kw in message_lower for kw in job_keywords) and not extracted.get("category"):
            return result
        
        result["has_job_query"] = True
        
        # Determine tool
        if extracted.get("job_id"):
            result["tool"] = "get_job_details"
            result["confidence"] = 0.95
        elif "công ty" in message_lower or "company" in message_lower:
            result["tool"] = "get_company_jobs"
            result["confidence"] = 0.85
        else:
            result["tool"] = "search_jobs"
            result["confidence"] = 0.75
        
        # Extract limit
        limit_match = re.search(r'limit\s*(\d+)|(\d+)\s*(?:jobs?|kết quả)', message_lower)
        if limit_match:
            result["limit"] = int(limit_match.group(1) or limit_match.group(2))
        
        # Extract salary range max if exists
        salary_range_match = re.search(r'(\d+)\s*(?:triệu|m)\s*-\s*(\d+)\s*(?:triệu|m)', message_lower)
        if salary_range_match:
            result["salary_max"] = f"{salary_range_match.group(2)}m"
        
        logger.info(f"[OK] Improved regex parsed: tool={result['tool']}, confidence={result['confidence']}")
        return result
    
    def _parse_message_with_regex(self, message: str) -> Dict[str, Any]:
        """Fast regex-based parsing (legacy fallback)"""
        # Use improved method as the new standard
        return self._parse_message_with_improved_regex(message)
    
    async def detect_job_intent(self, message: str) -> Optional[Dict[str, Any]]:
        """Detect job intent with Ollama + timeout + fallback"""
        parsed = await self._parse_message_with_ollama_timeout(message, timeout_sec=2.0)
        
        if not parsed.get("has_job_query"):
            return None
        
        tool_name = parsed.get("tool")
        if not tool_name:
            return None
        
        # Build params
        params = {}
        
        if tool_name == "search_jobs":
            if parsed.get("category"):
                params["category"] = parsed["category"]
            if parsed.get("location"):
                params["location"] = parsed["location"]
            if parsed.get("salary_min"):
                params["salary_min"] = parsed["salary_min"]
            
            params["query"] = "" if parsed.get("category") else (parsed.get("keywords") or "")
            params["limit"] = parsed.get("limit", 20)
            params["page"] = parsed.get("page", 1)
        
        elif tool_name == "get_job_details":
            if parsed.get("job_id"):
                params["job_id"] = parsed["job_id"]
            else:
                return None
        
        elif tool_name == "get_company_jobs":
            params["company_id"] = 1
            if parsed.get("company_name"):
                params["company_name"] = parsed["company_name"]
        
        logger.info(f"[INTENT] {tool_name}")
        return {"tool": tool_name, "params": params}
    
    async def call_job_tool_with_retry(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call backend API with retry logic"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"[CALL] {tool_name} attempt {attempt + 1}/{self.max_retries}")
                
                if tool_name == "search_jobs":
                    result = await asyncio.wait_for(
                        self.backend_client.search_jobs(
                            query=params.get("query", ""),
                            limit=params.get("limit", 100),
                            page=params.get("page", 1),
                            category=params.get("category"),
                            location=params.get("location"),
                            salary_min=params.get("salary_min")
                        ),
                        timeout=5.0
                    )
                
                elif tool_name == "get_job_details":
                    result = await asyncio.wait_for(
                        self.backend_client.get_job_details(job_id=params.get("job_id")), # type: ignore
                        timeout=5.0
                    )
                
                elif tool_name == "get_company_jobs":
                    result = await asyncio.wait_for(
                        self.backend_client.get_company_jobs(
                            company_id=params.get("company_id"), # type: ignore
                            page=params.get("page", 1),
                            limit=params.get("limit", 10)
                        ),
                        timeout=5.0
                    )
                else:
                    return {"error": f"Unknown tool: {tool_name}"}
                
                # Success
                logger.info(f"[OK] {tool_name} succeeded on attempt {attempt + 1}")
                print("\n" + "="*80)
                print(f"[BACKEND RESPONSE] {tool_name}:")
                print("="*80)
                print(json.dumps(result, ensure_ascii=False, indent=2)[:500])
                print("="*80 + "\n")
                return result
            
            except asyncio.TimeoutError:
                last_error = "timeout"
                logger.warning(f"[WARNING] {tool_name} timeout on attempt {attempt + 1}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                    continue
            
            except Exception as e:
                last_error = str(e)
                logger.warning(f"[WARNING] {tool_name} error on attempt {attempt + 1}: {str(e)[:50]}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                    continue
        
        # All retries failed
        logger.error(f"[ERROR] {tool_name} failed after {self.max_retries} attempts: {last_error}")
        return {"error": f"Failed after {self.max_retries} retries: {last_error}", "status": "failed"}
    

    def _format_job_detail(self, job: Dict[str, Any]) -> str:
        """Format chi tiết 1 job"""

        title = job.get("title", "N/A")
        company = job.get("name") or job.get("company_name") or "N/A"
        location = job.get("location") or job.get("workLocation") or "N/A"
        job_id = job.get("id", "N/A")

        description = job.get("description", "Không có mô tả")
        requirements = job.get("requirements", "")
        benefits = job.get("benefits", "")
        salary = job.get("salary", "")

        response = f"""
    🧑‍💻 {title}
    🆔 ID: {job_id}

    🏢 Công ty: {company}
    📍 Địa điểm: {location}
    💰 Lương: {salary if salary else "Thỏa thuận"}

    📄 MÔ TẢ CÔNG VIỆC:
    {description}

    """

        if requirements:
            response += f"\n📌 YÊU CẦU:\n{requirements}\n"

        if benefits:
            response += f"\n🎁 QUYỀN LỢI:\n{benefits}\n"

        return response



    def _format_job_results_for_chat(self, jobs_result: Dict[str, Any]) -> str:
        """Format job list hoặc job detail"""

        if not jobs_result:
            return "Không có dữ liệu."

        # ❗ Nếu lỗi
        if isinstance(jobs_result, dict) and "error" in jobs_result:
            return f"Lỗi: {jobs_result.get('error')}"

        # =========================
        # ✅ CASE: JOB DETAIL
        # =========================
        if isinstance(jobs_result, dict) and "id" in jobs_result:
            return self._format_job_detail(jobs_result)

        # =========================
        # ✅ CASE: JOB LIST
        # =========================
        jobs = jobs_result.get("jobs") or jobs_result.get("data") or []

        if not jobs:
            return "Không tìm thấy công việc phù hợp."

        response = f"🔎 Tìm thấy {len(jobs)} công việc:\n\n"

        for i, job in enumerate(jobs, 1):
            title = job.get("title") or job.get("job_title") or "N/A"
            company = job.get("company_name") or "N/A"
            location = job.get("location") or "N/A"
            job_id = job.get("id") or "N/A"

            response += f"{i}. {title} (ID: {job_id})\n"
            response += f"   🏢 {company}\n"
            response += f"   📍 {location}\n\n"

        response += "💡 Nhập ID để xem chi tiết"

        return response
    
    async def chat_with_tools(self, user_message: str, conversation_id: str, file_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Main chat handler with full retry + timeout protection"""
        try:
            logger.info(f"[CHAT] Processing: {user_message[:50]}")
            logger.info(f"[CHAT DEBUG] file_ids received: {file_ids} (type: {type(file_ids).__name__})")
            logger.info(f"[CHAT DEBUG] file_ids is None? {file_ids is None}, is empty? {not file_ids}")
            
            # If files (CVs) are attached, prioritize CV analysis
            if file_ids:
                logger.info(f"[INTENT] ✅ CV files attached: {file_ids}")
                logger.info(f"[INTENT] CV ANALYSIS MODE - calling cv_qa_simple")
                
                # Call simple CV Q&A tool (no embedding)
                from app.services.cv_qa_simple import answer_cv_question
                
                try:
                    cv_responses = []
                    for cv_id in file_ids:
                        logger.info(f"[CV QA] Processing CV: {cv_id}")
                        response = await answer_cv_question(cv_id, user_message)
                        cv_responses.append(response)
                    
                    # Combine responses
                    combined_response = "\n\n".join(cv_responses) if cv_responses else "Không thể phân tích CV. Vui lòng thử lại."
                    
                    # Convert file paths to source dictionaries for ConversationMessage model
                    sources_dicts = [{"type": "cv_file", "path": f} for f in file_ids]
                    
                    logger.info(f"[CV QA DONE] ✅ CV analysis complete, response length: {len(combined_response)}")
                    return {
                        "bot_response": combined_response,
                        "sources": sources_dicts,
                        "detected_intent": "cv_analysis",
                        "tool_used": "cv_qa_simple",
                    }
                except Exception as e:
                    logger.error(f"CV Q&A error: {e}")
                    import traceback
                    logger.error(f"[TRACEBACK] {traceback.format_exc()}")
                    return {
                        "bot_response": f"Lỗi trả lời câu hỏi CV: {str(e)[:100]}",
                        "sources": [],
                        "detected_intent": "cv_analysis_error",
                        "error": str(e)
                    }
            else:
                logger.info(f"[INTENT] ❌ No files attached, proceeding with job/RAG analysis")
            
            # Otherwise, check for job intent
            job_intent = await self.detect_job_intent(user_message)
            
            if job_intent:
                logger.info(f"[INTENT] Job search detected: {job_intent['tool']}")
                
                # Call with retry
                tool_result = await self.call_job_tool_with_retry(
                    job_intent["tool"],
                    job_intent["params"]
                )
                
                formatted_response = self._format_job_results_for_chat(tool_result)
                
                return {
                    "bot_response": formatted_response,
                    "sources": [],
                    "detected_intent": "job_search",
                    "tool_used": job_intent["tool"],
                    "tool_result": tool_result
                }
            
            else:
                logger.info("[OK] No job intent, using RAG")
                result = self.chatbot.chat_with_conversation(
                    user_message=user_message,
                    conversation_id=conversation_id,
                )
                return result
        
        except asyncio.TimeoutError:
            logger.error("[ERROR] Overall timeout in chat handler")
            return {
                "bot_response": "Xin lỗi, xử lý timed out. Vui lòng thử lại.",
                "sources": [],
                "detected_intent": "error",
                "error": "timeout"
            }
        except Exception as e:
            logger.error(f"[ERROR] Chat handler error: {e}")
            return {
                "bot_response": f"Lỗi: {str(e)[:100]}",
                "sources": [],
                "detected_intent": "error",
                "error": str(e)
            }
