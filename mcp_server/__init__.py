#!/usr/bin/env python3
"""
MCP Server - FastMCP Application
Model Context Protocol Server for AI CV-Job Matcher

Exposes backend APIs as tools for Claude and other AI models
Uses the modern FastMCP API for tool registration and handling
"""

import json
import logging
import os
import httpx
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import MCP
from mcp.server import FastMCP # type: ignore

# Import tools
from .tools import (
    CVTools, JobsTools, MatchingTools, ChatbotTools, CandidateTools,
    ALL_TOOLS
)

# Backend API configuration
BACKEND_BASE_URL = os.getenv("BACKEND_URL", "http://localhost:4000")
BACKEND_API_KEY = os.getenv("BACKEND_API_KEY", "")

# Create FastMCP server instance
mcp_server = FastMCP("ai-cv-matcher-mcp")

# HTTP client with connection pooling
http_client = httpx.AsyncClient(base_url=BACKEND_BASE_URL)


# ============================================================================
# TOOL INSTANCES
# ============================================================================

def get_headers():
    """Get request headers with API key if configured"""
    headers = {}
    if BACKEND_API_KEY:
        headers["X-API-Key"] = BACKEND_API_KEY
    return headers


def create_tool_instances():
    """Create instances of all tool classes"""
    headers = get_headers()
    
    return {
        "cv": CVTools(http_client, headers),
        "jobs": JobsTools(http_client, headers),
        "matching": MatchingTools(http_client, headers),
        "chatbot": ChatbotTools(http_client, headers),
        "candidate": CandidateTools(http_client, headers),
    }


tools = create_tool_instances()


# ============================================================================
# TOOL HANDLER
# ============================================================================

async def call_tool(name: str, arguments: dict) -> dict:
    """Route tool calls to appropriate handlers"""
    
    logger.info(f"Calling tool: {name} with args: {arguments}")
    
    try:
        # CV Tools
        if name == "analyze_cv":
            return await tools["cv"].analyze_cv(arguments["cv_text"])
        
        elif name == "get_cv_insights":
            return await tools["cv"].get_cv_insights(arguments["cv_id"])
        
        elif name == "generate_cv_improvements":
            return await tools["cv"].generate_cv_improvements(arguments["cv_id"])
        
        # Jobs Tools
        elif name == "search_jobs":
            return await tools["jobs"].search_jobs(
                arguments["query"],
                limit=arguments.get("limit", 20),
                category=arguments.get("category"),
                location=arguments.get("location"),
                salary_min=arguments.get("salary_min"),
                page=arguments.get("page", 1)
            )
        
        elif name == "get_job_details":
            return await tools["jobs"].get_job_details(arguments["job_id"])
        
        elif name == "list_all_jobs":
            return await tools["jobs"].list_all_jobs(
                page=arguments.get("page", 1),
                limit=arguments.get("limit", 20),
                active=arguments.get("active")
            )
        
        elif name == "get_company_jobs":
            return await tools["jobs"].get_company_jobs(
                arguments["company_id"],
                page=arguments.get("page", 1),
                limit=arguments.get("limit", 10),
                active=arguments.get("active")
            )
        
        # Matching Tools
        elif name == "match_cv_to_job":
            return await tools["matching"].match_cv_to_job(
                arguments["cv_id"],
                arguments["job_id"]
            )
        
        elif name == "find_best_matches":
            return await tools["matching"].find_best_matches(
                arguments["cv_id"],
                arguments.get("limit", 5)
            )
        
        elif name == "get_match_explanation":
            return await tools["matching"].get_match_explanation(
                arguments["cv_id"],
                arguments["job_id"]
            )
        
        # Chatbot Tools
        elif name == "chat":
            return await tools["chatbot"].chat(
                arguments["message"],
                arguments.get("conversation_id"),
                arguments.get("context_type", "general")
            )
        
        # Candidate Tools
        elif name == "create_candidate":
            return await tools["candidate"].create_candidate(
                arguments["name"],
                arguments["email"],
                arguments.get("phone"),
                arguments.get("skills")
            )
        
        elif name == "get_candidate":
            return await tools["candidate"].get_candidate(arguments["candidate_id"])
        
        elif name == "list_candidates":
            return await tools["candidate"].list_candidates(
                arguments.get("skip", 0),
                arguments.get("limit", 20),
                arguments.get("skill")
            )
        
        else:
            return {"error": f"Unknown tool: {name}"}
    
    except KeyError as e:
        return {"error": f"Missing required argument: {e}"}
    except Exception as e:
        logger.error(f"Error calling tool {name}: {e}")
        return {"error": str(e), "status": "failed"}


# ============================================================================
# REGISTER TOOLS WITH FASTMCP
# ============================================================================

# Define wrapper functions for each tool that FastMCP can wrap
@mcp_server.tool()
async def analyze_cv(cv_text: str) -> dict:
    """Analyze a CV file and extract key information"""
    return await call_tool("analyze_cv", {"cv_text": cv_text})

@mcp_server.tool()
async def get_cv_insights(cv_id: str) -> dict:
    """Get AI-generated insights about a CV"""
    return await call_tool("get_cv_insights", {"cv_id": cv_id})

@mcp_server.tool()
async def generate_cv_improvements(cv_id: str) -> dict:
    """Generate specific improvement suggestions for a CV"""
    return await call_tool("generate_cv_improvements", {"cv_id": cv_id})

@mcp_server.tool()
async def search_jobs(
    query: str,
    limit: int = 20,
    page: int = 1,
    category: str = None,
    location: str = None,
    salary_min: str = None
) -> dict:
    """Search for jobs by keywords, skills, location, category, and salary range"""
    return await call_tool("search_jobs", {
        "query": query,
        "limit": limit,
        "page": page,
        "category": category,
        "location": location,
        "salary_min": salary_min
    })

@mcp_server.tool()
async def get_job_details(job_id: int) -> dict:
    """Get detailed information about a specific job posting"""
    return await call_tool("get_job_details", {"job_id": job_id})

@mcp_server.tool()
async def list_all_jobs(page: int = 1, limit: int = 20, active: bool = None) -> dict:
    """List all available jobs with pagination and optional active status filter"""
    return await call_tool("list_all_jobs", {
        "page": page,
        "limit": limit,
        "active": active
    })

@mcp_server.tool()
async def get_company_jobs(
    company_id: int,
    page: int = 1,
    limit: int = 10,
    active: bool = None
) -> dict:
    """Get all jobs posted by a specific company"""
    return await call_tool("get_company_jobs", {
        "company_id": company_id,
        "page": page,
        "limit": limit,
        "active": active
    })

@mcp_server.tool()
async def match_cv_to_job(cv_id: str, job_id: str) -> dict:
    """Match a CV against a job posting and get compatibility score"""
    return await call_tool("match_cv_to_job", {"cv_id": cv_id, "job_id": job_id})

@mcp_server.tool()
async def find_best_matches(cv_id: str, limit: int = 5) -> dict:
    """Find the best job matches for a given CV"""
    return await call_tool("find_best_matches", {"cv_id": cv_id, "limit": limit})

@mcp_server.tool()
async def get_match_explanation(cv_id: str, job_id: str) -> dict:
    """Get detailed explanation of why a CV matches a job"""
    return await call_tool("get_match_explanation", {"cv_id": cv_id, "job_id": job_id})

@mcp_server.tool()
async def chat(message: str, conversation_id: str = None, context_type: str = "general") -> dict:
    """Send a message to the AI chatbot for advice and assistance"""
    return await call_tool("chat", {
        "message": message,
        "conversation_id": conversation_id,
        "context_type": context_type
    })

@mcp_server.tool()
async def create_candidate(name: str, email: str, phone: str = None, skills: list = None) -> dict:
    """Create a new candidate profile"""
    return await call_tool("create_candidate", {
        "name": name,
        "email": email,
        "phone": phone,
        "skills": skills
    })

@mcp_server.tool()
async def get_candidate(candidate_id: str) -> dict:
    """Get candidate profile information"""
    return await call_tool("get_candidate", {"candidate_id": candidate_id})

@mcp_server.tool()
async def list_candidates(skip: int = 0, limit: int = 20, skill: str = None) -> dict:
    """List all candidates with optional filtering"""
    return await call_tool("list_candidates", {
        "skip": skip,
        "limit": limit,
        "skill": skill
    })


# Log tool registration
logger.info(f"Registered {len(ALL_TOOLS)} MCP tools")
for tool in ALL_TOOLS:
    logger.debug(f"  ✓ {tool['name']}: {tool['description']}")


# ============================================================================
# MAIN
# ============================================================================

async def main():
    """Start MCP server"""
    logger.info("Starting MCP Server")
    logger.info(f"Backend URL: {BACKEND_BASE_URL}")
    logger.info(f"Total Tools: {len(ALL_TOOLS)}")
    
    # List available tools by category
    logger.info("\nAvailable Tools:")
    logger.info("=" * 70)
    logger.info("CV Analysis (3 tools):")
    for tool in ALL_TOOLS[:3]:
        logger.info(f"  ✓ {tool['name']}: {tool['description']}")
    
    logger.info("\nJobs (3 tools):")
    for tool in ALL_TOOLS[3:6]:
        logger.info(f"  ✓ {tool['name']}: {tool['description']}")
    
    logger.info("\nMatching (3 tools):")
    for tool in ALL_TOOLS[6:9]:
        logger.info(f"  ✓ {tool['name']}: {tool['description']}")
    
    logger.info("\nChatbot (1 tool):")
    for tool in ALL_TOOLS[9:10]:
        logger.info(f"  ✓ {tool['name']}: {tool['description']}")
    
    logger.info("\nCandidate Management (3 tools):")
    for tool in ALL_TOOLS[10:13]:
        logger.info(f"  ✓ {tool['name']}: {tool['description']}")
    
    logger.info("=" * 70)
    
    # Run server using stdio transport (default for MCP)
    await mcp_server.run_stdio_async()


def run():
    """Entry point for running MCP server"""
    import asyncio
    asyncio.run(main())


if __name__ == "__main__":
    run()
