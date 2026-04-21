"""
MCP Tools Package
Provides all available MCP tools for the server

Structure:
  - cv_tools: CV analysis and improvement
  - jobs_tools: Job search and listing
  - matching_tools: CV-to-job matching
  - chatbot_tools: AI chatbot interactions
  - candidate_tools: Candidate management
"""

from .cv_tools import CVTools, CV_TOOLS
from .jobs_tools import JobsTools, JOBS_TOOLS
from .matching_tools import MatchingTools, MATCHING_TOOLS
from .chatbot_tools import ChatbotTools, CHATBOT_TOOLS
from .candidate_tools import CandidateTools, CANDIDATE_TOOLS

__all__ = [
    # Tool classes
    "CVTools",
    "JobsTools",
    "MatchingTools",
    "ChatbotTools",
    "CandidateTools",
    
    # Tool definitions
    "CV_TOOLS",
    "JOBS_TOOLS",
    "MATCHING_TOOLS",
    "CHATBOT_TOOLS",
    "CANDIDATE_TOOLS",
]

# Combined list of all tools
ALL_TOOLS = [
    *CV_TOOLS,
    *JOBS_TOOLS,
    *MATCHING_TOOLS,
    *CHATBOT_TOOLS,
    *CANDIDATE_TOOLS,
]
