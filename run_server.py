#!/usr/bin/env python3
"""
Simple script to start the FastAPI server with proper imports
"""
import subprocess
import sys
import os
import time

# Add project root to Python path so 'app' module can be imported
os.environ['PYTHONPATH'] = os.getcwd()

# Small delay to ensure previous process fully released the port
time.sleep(1)

# Run uvicorn on port 8001 with reuse_port option
subprocess.run(
    [sys.executable, '-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', '8001', '--reload'],
    cwd=os.getcwd()
)
