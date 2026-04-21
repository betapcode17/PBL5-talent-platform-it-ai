#!/usr/bin/env python3
"""Test MongoDB Atlas connection"""

import os
from dotenv import load_dotenv
from pathlib import Path
from pymongo import MongoClient

# Load .env
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "cv_job_matcher")

print("Testing MongoDB Atlas Connection...")
print(f"URL: {MONGODB_URL[:50]}...")
print(f"Database: {MONGODB_DATABASE}")

try:
    client = MongoClient(MONGODB_URL, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    print("✓ Connected to MongoDB Atlas!")
    
    db = client[MONGODB_DATABASE]
    print(f"✓ Using database: {MONGODB_DATABASE}")
    
    # Test collections
    collections = db.list_collection_names()
    print(f"✓ Collections: {collections if collections else 'None (will be created on first insert)'}")
    
    # Test insert and find
    test_doc = {"test": "connection"}
    result = db["test_collection"].insert_one(test_doc)
    print(f"✓ Test insert successful: {result.inserted_id}")
    
    # Clean up test
    db["test_collection"].delete_one({"_id": result.inserted_id})
    print("✓ Test cleanup successful")
    
    print("\n✓✓✓ MongoDB Atlas is ready to use! ✓✓✓")
    
except Exception as e:
    print(f"✗ Connection failed: {e}")
    print("\nMake sure:")
    print("  1. MONGODB_URL in .env is correct")
    print("  2. MongoDB Atlas cluster is running")
    print("  3. Your IP is whitelisted in MongoDB Atlas (Network Access)")
    print("  4. Database user has proper permissions")
