#!/usr/bin/env python3
"""
Test MongoDB connection and setup guide.
"""

import sys
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

# Test local MongoDB
print("=" * 60)
print("Testing MongoDB Connection")
print("=" * 60)

# Test 1: Local MongoDB
print("\n[1] Testing LOCAL MongoDB (localhost:27017)...")
try:
    client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=3000)
    client.admin.command('ping')
    print("✓ Connected to LOCAL MongoDB")
    print("  Use this URL: mongodb://localhost:27017")
    sys.exit(0)
except (ConnectionFailure, ServerSelectionTimeoutError) as e:
    print(f"✗ LOCAL MongoDB not available: {e}")

# Test 2: MongoDB Atlas (Cloud)
print("\n[2] MongoDB Atlas Setup Guide:")
print("  1. Go to: https://www.mongodb.com/cloud/atlas")
print("  2. Create account (FREE tier available)")
print("  3. Create cluster")
print("  4. Get connection string like:")
print("     mongodb+srv://username:password@cluster.mongodb.net/database")
print("\n  Then set environment variable:")
print("     export MONGODB_URL='mongodb+srv://user:pass@cluster.mongodb.net/cv_job_matcher'")

# Test 3: Local MongoDB Install Guide
print("\n[3] Install LOCAL MongoDB Community:")
print("  Windows: https://www.mongodb.com/try/download/community")
print("  macOS:   brew install mongodb-community")
print("  Linux:   sudo apt install mongodb")
print("\n  After install, start service:")
print("    macOS/Linux: brew services start mongodb-community")
print("    Windows:     mongod")

print("\n" + "=" * 60)
print("Current settings in config.py:")
print("  MONGODB_URL = 'mongodb://localhost:27017'")
print("  MONGODB_DATABASE = 'cv_job_matcher'")
print("=" * 60)
