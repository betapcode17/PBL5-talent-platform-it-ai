#!/usr/bin/env python3
"""
Simple script to start the FastAPI server with proper imports
"""
import subprocess
import sys
import os

# Add app directory to Python path
os.environ['PYTHONPATH'] = os.path.join(os.getcwd(), 'app')

# Run uvicorn on port 8001 (fallback if 8000 is busy)
subprocess.run(
    [sys.executable, '-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', '8001'],
    cwd=os.getcwd()
)
