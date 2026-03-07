# app/services/llm_service.py
"""
LLM Service - Menggunakan Google Gemini Chat API untuk generate responses.
Handles LLM calls, temperature settings, dan response formatting.
"""

import logging
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from services.api_key_manager import get_next_api_key

logger = logging.getLogger(__name__)


class LLMService:
    """Service untuk interaksi dengan Gemini Chat API"""
    
    def __init__(
        self, 
        model_name: str = "gemini-2.0-flash-lite",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        """
        Initialize LLM Service
        
        Args:
            model_name: Model Gemini yang digunakan (gemini-2.5-flash, gemini-2.5-flash-vision)
            temperature: Creativity level (0.0-1.0, default 0.7)
            max_tokens: Max response length
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self) -> ChatGoogleGenerativeAI:
        """Initialize Gemini Chat LLM"""
        try:
            api_key = get_next_api_key()
            if not api_key:
                raise ValueError(" GOOGLE_API_KEY not found")
            
            llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=api_key,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                convert_system_message_to_human=True,
                timeout=30
            )
            logger.info(f" Initialized {self.model_name} with temperature={self.temperature}")
            return llm
        except Exception as e:
            logger.error(f" Failed to initialize LLM: {e}")
            raise
    
    def generate_response(
        self, 
        user_message: str, 
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate response from user message
        
        Args:
            user_message: User's question/input
            system_prompt: Optional system instruction
            temperature: Optional temperature override
            
        Returns:
            LLM response text
        """
        try:
            # Build messages
            messages = []
            
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            
            messages.append(HumanMessage(content=user_message))
            
            # Use custom temperature if provided
            if temperature is not None and temperature != self.temperature:
                # Create new LLM with custom temp (fallback to default LLM if needed)
                self.llm.temperature = temperature
            
            # Get response
            response = self.llm.invoke(messages)
            
            # Reset temperature if changed
            if temperature is not None:
                self.llm.temperature = self.temperature
            
            return response.content # type: ignore
            
        except Exception as e:
            logger.error(f" Error generating response: {e}")
            raise
    
    def generate_with_context(
        self,
        user_message: str,
        context: str,
        system_prompt: str
    ) -> str:
        """
        Generate response with RAG context
        
        Args:
            user_message: User question
            context: Retrieved context from RAG
            system_prompt: System instructions
            
        Returns:
            Response text
        """
        # Combine context with user message
        full_message = f"""
Context từ tài liệu:
{context}

Câu hỏi của người dùng:
{user_message}

Hãy trả lời dựa trên context trên. Nếu context không đủ, hãy nói rõ.
"""
        return self.generate_response(full_message, system_prompt)
    
    def extract_entities(self, text: str) -> dict:
        """
        Extract job-related entities from text
        (job titles, locations, skills, salaries, etc)
        """
        extraction_prompt = f"""
Hãy trích xuất các thông tin sau từ text:
- Job titles (chức vị)
- Locations (địa điểm)
- Skills (kỹ năng)
- Salary range (mức lương)
- Experience level (mức độ kinh nghiệm)
- Company names (tên công ty)

Text: {text}

Trả lời dưới dạng JSON:
{{
    "job_titles": [],
    "locations": [],
    "skills": [],
    "salary_range": null,
    "experience_level": null,
    "company_names": []
}}
"""
        try:
            response = self.generate_response(extraction_prompt)
            # Parse JSON response
            import json
            # Find JSON in response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start < end:
                json_str = response[start:end]
                return json.loads(json_str)
            return {}
        except Exception as e:
            logger.warning(f" Entity extraction failed: {e}")
            return {}
    
    def summarize(self, text: str, max_length: int = 200) -> str:
        """
        Summarize text about job
        """
        prompt = f"""
Hãy tóm tắt nội dung sau thành {max_length} ký tự:

{text}

Tóm tắt:"""
        return self.generate_response(prompt)
    
    def rate_match(self, cv_summary: str, job_description: str) -> dict:
        """
        Rate CV-Job match compatibility
        """
        prompt = f"""
CV:
{cv_summary}

Job Description:
{job_description}

Hãy đánh giá mức độ phù hợp (0-100%) giữa CV và Job Description.
Trả lời dưới dạng JSON:
{{
    "match_percentage": 0-100,
    "strengths": [],
    "gaps": [],
    "recommendation": "Strongly Recommended / Recommended / Consider / Not Recommended"
}}
"""
        try:
            response = self.generate_response(prompt)
            import json
            start = response.find('{')
            end = response.rfind('}') + 1
            if start < end:
                json_str = response[start:end]
                return json.loads(json_str)
            return {"match_percentage": 0, "recommendation": "Error"}
        except Exception as e:
            logger.error(f" Match rating failed: {e}")
            return {"match_percentage": 0, "recommendation": "Error"}


# Global LLM instance
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get or create global LLM service instance"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


def reset_llm_service():
    """Reset LLM service (for testing)"""
    global _llm_service
    _llm_service = None
