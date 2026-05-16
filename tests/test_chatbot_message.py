import json
import asyncio
from fastapi.testclient import TestClient

from app.main import app


class FakeChatbot:
    def __init__(self, jobs=None):
        self._jobs = jobs or []

    async def chat(self, user_message, conversation_id, extra_context=None, retrieval_profile=None):
        total = len(self._jobs)
        data = {"jobs": self._jobs, "total": total}
        message_content = "Chi tiet ket qua" if total > 0 else None
        return {
            "success": True,
            "version": "2.1.0",
            "data": data,
            "message": {"content": message_content} if message_content else {},
            "retrieval": {"profile": "balanced", "count": total, "topScore": 1.0 if total else 0.0, "fallbackTriggered": False},
            "meta": {
                "latencyMs": None,
                "retrieval": {"latencyMs": None, "rerankLatencyMs": None, "contextPackingLatencyMs": None},
                "generation": {"model": None, "latencyMs": None, "promptTokens": None, "completionTokens": None},
            },
        }

    def health(self):
        return {"status": "ok"}


def setup_fake_chatbot(client, fake_instance):
    # inject the fake singleton used by router get_chatbot()
    import app.services.chatbot_service as cs

    cs._chatbot_singleton = fake_instance


def test_send_message_returns_jobs_and_stores_conversation():
    client = TestClient(app)

    fake = FakeChatbot(jobs=[{"id": "1", "title": "Backend Developer", "company": "ACME", "location": "HCM", "salary": "10-20m", "skills": ["python"], "score": 0.92, "url": "http://example"}])
    setup_fake_chatbot(client, fake)

    payload = {"message": "Tim job backend"}
    resp = client.post("/chatbot/message", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True
    assert "conversationId" in body and body["conversationId"]
    # message content stored in conversation should include the prefix when jobs found
    msg = body["message"]
    assert "content" in msg and "Tim thay" in (msg.get("content") or "")
    # data.totalJobsFound should equal 1
    assert body["data"]["totalJobsFound"] == 1


def test_send_message_no_jobs_returns_empty_data():
    client = TestClient(app)
    fake = FakeChatbot(jobs=[])
    setup_fake_chatbot(client, fake)

    payload = {"message": "Tao muon gi do khong lien quan"}
    resp = client.post("/chatbot/message", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["totalJobsFound"] == 0
