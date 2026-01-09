"""
Enum models for AI CV-Job Matcher.
"""

from enum import Enum

class ModelName(str, Enum):
    gemini_2_5_flash = "gemini-2.5-flash"
    gemini_2_0_flash_exp = "gemini-2.0-flash-exp"