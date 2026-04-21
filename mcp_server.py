#!/usr/bin/env python3
"""
MCP Server Entry Point
Simplified entry point that uses the refactored mcp_server package

The actual implementation is in mcp_server/ directory for better organization
"""

from mcp_server import run

if __name__ == "__main__":
    run()

