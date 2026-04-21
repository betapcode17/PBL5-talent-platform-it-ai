# app/services/llm_service.py
"""
LLM Service - Local Ollama integration
Uses Ollama for all LLM operations (local, no API key required)
"""

import logging
from typing import Optional
from app.config import LLM_PROVIDER, OLLAMA_BASE_URL, OLLAMA_MODEL

logger = logging.getLogger(__name__)


class LLMService:
    """Unified LLM Service supporting multiple backends"""
    
    def __init__(
        self, 
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ):
        """
        Initialize LLM Service with Ollama backend
        
        Args:
            provider: Ignored - always uses "ollama"
            model_name: Model name to use (from config or parameter)
            temperature: Creativity level (0.0-1.0)
            max_tokens: Max response length
        """
        from .ollama_service import OllamaService
        
        self.provider = "ollama"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_name = model_name or OLLAMA_MODEL
        
        self.llm = OllamaService(
            base_url=OLLAMA_BASE_URL,
            model_name=self.model_name,
            temperature=temperature,
            **kwargs
        )
        logger.info(f"[OK] Using Ollama backend ({self.model_name})")
    
    def generate_response(
        self, 
        user_message: str, 
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout_override: Optional[int] = None
    ) -> str:
        """
        Generate response from user message
        
        Args:
            user_message: User's question/input
            system_prompt: Optional system instruction
            temperature: Optional temperature override
            timeout_override: Override default timeout (useful for long operations like CV analysis)
            
        Returns:
            LLM response text
        """
        try:
            return self.llm.generate_response(
                user_message,
                system_prompt=system_prompt,
                temperature=temperature,
                timeout_override=timeout_override
            )
        except Exception as e:
            logger.error(f"✗ Error generating response: {e}")
            raise
    
    def summarize(self, text: str, max_length: int = 200) -> str:
        """Summarize text about job"""
        return self.llm.summarize(text, max_length)
    
    def rate_match(self, cv_summary: str, job_description: str) -> dict:
        """Rate CV-Job match compatibility"""
        return self.llm.rate_match(cv_summary, job_description)

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

