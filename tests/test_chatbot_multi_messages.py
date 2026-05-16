import time
from fastapi.testclient import TestClient

from app.main import app


class MultiFakeChatbot:
    def __init__(self, jobs=None):
        # jobs can be a list of job dicts or a mapping by keyword
        self._jobs = jobs or []

    async def chat(self, user_message, conversation_id, extra_context=None, retrieval_profile=None):
        # simple behavior: return all jobs regardless of query
        total = len(self._jobs)
        data = {"jobs": self._jobs, "total": total}
        message_content = f"Tìm thấy {total} công việc" if total > 0 else None
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


def setup_fake_chatbot(fake_instance):
    import app.services.chatbot_service as cs

    cs._chatbot_singleton = fake_instance


def test_multiple_messages_flow():
    client = TestClient(app)

    # create a bigger job list to simulate many results
    jobs = [
        {"id": str(i), "title": f"Developer {i}", "company": "ACME", "location": "HCM", "salary": "10-20m", "skills": ["python"], "score": 0.9, "url": "http://example"}
        for i in range(1, 11)
    ]

    fake = MultiFakeChatbot(jobs=jobs)
    setup_fake_chatbot(fake)

    # simulate a user sending multiple queries in the same conversation
    conversation_id = None
    queries = [
        "Tìm job backend",
        "Tìm job frontend",
        "Tìm job data",
        "Cho tôi những job remote",
        "Có job senior không?",
    ]

    for q in queries:
        payload = {"message": q}
        resp = client.post("/chatbot/message", json=payload)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["success"] is True
        # check that totalJobsFound reflects our fake jobs
        assert body["data"]["totalJobsFound"] == len(jobs)
        # first response should include a conversationId
        if not conversation_id:
            conversation_id = body.get("conversationId")
            assert conversation_id
        # small delay to mimic user typing
        time.sleep(0.05)
