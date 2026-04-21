"""
Chatbot Tools
Handles AI-powered chatbot interactions
"""

import logging
from typing import Optional
import httpx

logger = logging.getLogger(__name__)


class ChatbotTools:
    """AI chatbot tools"""
    
    def __init__(self, http_client: httpx.AsyncClient, headers: dict):
        self.http_client = http_client
        self.headers = headers
    
    async def chat(self, message: str, conversation_id: Optional[str] = None, 
                   context_type: str = "general") -> dict:
        """Send a message to the AI chatbot for advice and assistance"""
        try:
            response = await self.http_client.post(
                "/chatbot/message",
                json={
                    "message": message,
                    "conversation_id": conversation_id,
                    "context_type": context_type
                },
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error sending chat message: {e}")
            return {"error": str(e), "status": "failed"}


# Tool definitions for registration
CHATBOT_TOOLS = [
    {
        "name": "chat",
        "description": "Send a message to the AI chatbot for advice and assistance",
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The user message or question"
                },
                "conversation_id": {
                    "type": "string",
                    "description": "Optional conversation ID for context continuity"
                },
                "context_type": {
                    "type": "string",
                    "enum": ["cv", "jobs", "matching", "career", "general"],
                    "description": "Type of context for the conversation"
                }
            },
            "required": ["message"]
        }
    },
]
