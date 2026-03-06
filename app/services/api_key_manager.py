"""
API Key Manager - Rotate between multiple Google API keys to avoid quota limits
"""
import os
import logging
from typing import List
import random

class APIKeyManager:
    """
    Manages multiple Google API keys with rotation strategy
    to avoid hitting quota limits (50 requests/day for free tier)
    """
    
    def __init__(self):
        """Initialize with API keys from environment variables"""
        self.api_keys: List[str] = []
        self.current_index = 0
        
        # Load all API keys from .env
        for i in range(1, 10):  # Support up to 9 keys
            key = os.getenv(f"GOOGLE_API_KEY_{i}")
            if key:
                self.api_keys.append(key)
                logging.info(f" Loaded API key {i}: {key[:20]}...")
        
        if not self.api_keys:
            # Fallback to single key
            fallback_key = os.getenv("GOOGLE_API_KEY")
            if fallback_key:
                self.api_keys.append(fallback_key)
                logging.warning(" Using single API key (no rotation)")
            else:
                raise ValueError(" No Google API keys found in .env file!")
        
        logging.info(f" API Key Manager initialized with {len(self.api_keys)} keys")
        
        # Randomize starting index to distribute load
        self.current_index = random.randint(0, len(self.api_keys) - 1)
    
    def get_next_key(self) -> str:
        """
        Get next API key using round-robin rotation
        
        Returns:
            str: Next API key in rotation
        """
        key = self.api_keys[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.api_keys)
        
        logging.debug(f" Using API key {self.current_index + 1}/{len(self.api_keys)}")
        return key
    
    def get_random_key(self) -> str:
        """
        Get random API key (useful for parallel requests)
        
        Returns:
            str: Random API key
        """
        key = random.choice(self.api_keys)
        logging.debug(f" Using random API key")
        return key
    
    def get_all_keys(self) -> List[str]:
        """
        Get all available API keys
        
        Returns:
            List[str]: All API keys
        """
        return self.api_keys.copy()
    
    def get_key_count(self) -> int:
        """
        Get number of available API keys
        
        Returns:
            int: Number of keys
        """
        return len(self.api_keys)


# Global instance
_api_key_manager = None

def get_api_key_manager() -> APIKeyManager:
    """
    Get global API key manager instance (singleton pattern)
    
    Returns:
        APIKeyManager: Global instance
    """
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager()
    return _api_key_manager


def get_next_api_key() -> str:
    """
    Convenience function to get next API key
    
    Returns:
        str: Next API key in rotation
    """
    return get_api_key_manager().get_next_key()


def get_random_api_key() -> str:
    """
    Convenience function to get random API key
    
    Returns:
        str: Random API key
    """
    return get_api_key_manager().get_random_key()
