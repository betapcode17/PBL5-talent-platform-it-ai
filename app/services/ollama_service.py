"""
Ollama Service - Local LLM using Ollama
Replaces Google Gemini with local Ollama models
Supports streaming and non-streaming generation
Supports both sync and async operations
"""

import logging
import json
import requests
import asyncio
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Thread pool for running sync operations in async context
_executor = ThreadPoolExecutor(max_workers=5)


class OllamaService:
    """Service untuk interaksi dengan Ollama local LLM"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "qwen2.5:3b",
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        num_ctx: int = 2048,
        timeout: int = 180,
    ):
        """
        Initialize Ollama Service
        
        Args:
            base_url: Ollama API endpoint (default: http://localhost:11434)
            model_name: Model name to use (e.g., qwen2.5:3b, llama2, mistral)
            temperature: Creativity level (0.0-1.0, default 0.7)
            top_p: Nucleus sampling (0.0-1.0)
            top_k: Top-K sampling
            num_ctx: Context window size
            timeout: Request timeout in seconds (120s for qwen2.5:3b which needs ~18-20s)
        """
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.num_ctx = num_ctx
        self.timeout = timeout
        
        # Test connection
        self._test_connection()
        logger.info(f"[OK] Ollama Service initialized ({self.model_name}, timeout={timeout}s)")
    
    def _test_connection(self) -> bool:
        """Test connection to Ollama server"""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            response.raise_for_status()
            logger.info(f"✓ Connected to Ollama at {self.base_url}")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to connect to Ollama: {e}")
            raise
    
    def generate_response(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        timeout_override: Optional[int] = None
    ) -> str:
        """
        Generate response from user message using Ollama
        
        Args:
            user_message: User's question/input
            system_prompt: Optional system instruction
            temperature: Optional temperature override
            stream: Whether to stream response
            timeout_override: Override default timeout (in seconds)
            
        Returns:
            LLM response text
        """
        try:
            # Use custom temperature if provided
            temp = temperature if temperature is not None else self.temperature
            
            # Use custom timeout if provided, otherwise use default
            timeout = timeout_override if timeout_override is not None else self.timeout
            
            # Build prompt with system context
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{user_message}"
            else:
                full_prompt = user_message
            
            # Prepare request payload
            payload = {
                "model": self.model_name,
                "prompt": full_prompt,
                "stream": stream,
                "options": {
                    "temperature": temp,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "num_ctx": self.num_ctx,
                }
            }
            
            logger.info(f"[OLLAMA] Sending request with timeout={timeout}s, model={self.model_name}")
            
            # Make request to Ollama
            # Note: timeout tuple is (connect_timeout, read_timeout)
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=(10, timeout),  # 10s connect, timeout seconds for read
                stream=stream
            )
            response.raise_for_status()
            
            # Handle response
            if stream:
                # For streaming, collect all chunks
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            if "response" in chunk:
                                full_response += chunk["response"]
                        except json.JSONDecodeError:
                            continue
                return full_response.strip()
            else:
                # Non-streaming response
                result = response.json()
                if "response" in result:
                    return result["response"].strip()
                else:
                    raise ValueError("Unexpected Ollama response format")
                    
        except requests.exceptions.Timeout as e:
            logger.error(f"[OLLAMA] ✗ TIMEOUT after {timeout}s: {str(e)}")
            raise
        except requests.exceptions.ConnectionError as e:
            logger.error(f"[OLLAMA] ✗ Connection to Ollama failed: {e}")
            raise
        except Exception as e:
            logger.error(f"[OLLAMA] ✗ Error generating response: {type(e).__name__}: {e}")
            raise
    
    async def ainvoke(self, content: str) -> object:
        """
        Async version for compatibility with LangChain interface
        Used by ai_analysis.py and other async code
        
        Args:
            content: Prompt text
            
        Returns:
            Object with .content attribute (compatible with LangChain)
        """
        response = await asyncio.get_event_loop().run_in_executor(
            _executor,
            self.generate_response,
            content,
            None,  # system_prompt
            None   # temperature
        )
        
        # Return object compatible with LangChain response format
        class Response:
            def __init__(self, text):
                self.content = text
        
        return Response(response)
    
    async def agenerate_response(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Async version of generate_response
        """
        return await asyncio.get_event_loop().run_in_executor(
            _executor,
            self.generate_response,
            user_message,
            system_prompt,
            temperature
        )
    
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
            logger.warning(f"✗ Entity extraction failed: {e}")
            return {}
    
    def summarize(self, text: str, max_length: int = 200) -> str:
        """
        Summarize text about job
        """
        prompt = f"""
Hãy tóm tắt nội dung sau thành {max_length} ký tự:

{text}

Tóm tắt:"""
        try:
            return self.generate_response(prompt)
        except Exception as e:
            logger.error(f"✗ Summarization failed: {e}")
            return ""
    
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
            logger.error(f"✗ Match rating failed: {e}")
            return {"match_percentage": 0, "recommendation": "Error"}


# Global Ollama instance
_ollama_service: Optional[OllamaService] = None


def get_ollama_service() -> OllamaService:
    """Get or create global Ollama service instance"""
    global _ollama_service
    if _ollama_service is None:
        from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL
        _ollama_service = OllamaService(
            base_url=OLLAMA_BASE_URL,
            model_name=OLLAMA_MODEL
        )
    return _ollama_service


def reset_ollama_service():
    """Reset Ollama service (for testing)"""
    global _ollama_service
    _ollama_service = None
