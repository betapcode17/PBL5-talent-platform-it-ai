#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Test - Test tất cả 16 messages qua /chatbot/message
"""
import asyncio
import aiohttp
import sys
from typing import Dict, Any

# Fix encoding
if sys.stdout.encoding.lower() in ['cp1252', 'ascii']:
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)

BASE_URL = "http://localhost:8001"

ALL_MESSAGES = [
    "Find Java jobs in Ho Chi Minh City",
    "Any remote positions for DevOps?",
    "Jobs with salary over 30 million in Da Nang?",
    "Where are Senior Python Developer openings?",
    "What programming languages are hottest right now?",
    "Average salary for Frontend Developer?",
    "Compare Java vs Python in the VN market?",
    "IT recruitment trends in 2026?",
    "What is my CV missing for a Backend role?",
    "How to write a CV for IT freshers?",
    "Which skills should I highlight in my CV?",
    "English or Vietnamese CV — which is better?",
    "I know Python, what should I learn next?",
    "Roadmap from Junior to Senior Developer?",
    "Fullstack or specialized Backend?",
    "Should I switch from Tester to Dev?",
]

CATEGORIES = {
    "🔍 Job Search": ALL_MESSAGES[0:4],
    "📊 Market Analysis": ALL_MESSAGES[4:8],
    "📄 CV Advice": ALL_MESSAGES[8:12],
    "🎯 Career Advice": ALL_MESSAGES[12:16],
}


async def test_message(session, message):
    """Test một message"""
    url = f"{BASE_URL}/chatbot/message"
    try:
        timeout = aiohttp.ClientTimeout(total=60)
        async with session.post(url, json={"message": message}, timeout=timeout) as resp:
            if resp.status == 200:
                data = await resp.json()
                return {"ok": True, "msg": message[:50]}
            else:
                return {"ok": False, "msg": message[:50], "err": f"HTTP {resp.status}"}
    except asyncio.TimeoutError:
        return {"ok": False, "msg": message[:50], "err": "Timeout"}
    except Exception as e:
        return {"ok": False, "msg": message[:50], "err": str(e)[:50]}


async def main():
    print("=" * 80)
    print("🧪 TEST CHATBOT - 16 MESSAGES")
    print("=" * 80)
    print()
    
    async with aiohttp.ClientSession() as session:
        # Check API
        try:
            async with session.get(f"{BASE_URL}/chatbot/health", timeout=5) as r:
                if r.status != 200:
                    print(f"❌ API not responding: HTTP {r.status}\n")
                    return
                print(f"✅ API running at {BASE_URL}\n")
        except Exception as e:
            print(f"❌ Cannot connect: {e}")
            print(f"   Run: python run_server.py\n")
            return
        
        total = 0
        passed = 0
        
        for cat, msgs in CATEGORIES.items():
            print(f"{cat}")
            print("-" * 80)
            
            for msg in msgs:
                result = await test_message(session, msg)
                total += 1
                
                if result["ok"]:
                    passed += 1
                    print(f"  ✅ OK | {result['msg']}")
                else:
                    print(f"  ❌ FAIL | {result['msg']} | {result['err']}")
            
            print()
        
        # Summary
        print("=" * 80)
        print(f"TOTAL: {total} | PASSED: {passed} | FAILED: {total - passed}")
        print(f"SUCCESS: {(passed*100//total) if total > 0 else 0}%")
        print("=" * 80)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n❌ Test interrupted")
    except Exception as e:
        print(f"\n\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()
