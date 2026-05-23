import asyncio


def test_analyze_cv_against_jobs_derives_learning_suggestions(monkeypatch):
    import app.services.ai_analysis as ai_analysis

    class FakeDoc:
        def __init__(self, metadata):
            self.metadata = metadata

    class FakeVectorStore:
        def similarity_search_with_relevance_scores(self, query, k):
            return [
                (
                    FakeDoc(
                        {
                            "job_id": 1,
                            "job_title": "AI Engineer",
                            "company": "ACME AI",
                            "skills": "Python, Machine Learning, PyTorch",
                            "salary": "20-30m",
                            "location": "HCM",
                            "work_type": "Full-time",
                        }
                    ),
                    0.95,
                ),
                (
                    FakeDoc(
                        {
                            "job_id": 2,
                            "job_title": "ML Engineer",
                            "company": "Beta Labs",
                            "skills": "Python, Docker, Kubernetes",
                            "salary": "25-35m",
                            "location": "Hanoi",
                            "work_type": "Full-time",
                        }
                    ),
                    0.89,
                ),
            ]

        def similarity_search(self, query, k):
            return []

    monkeypatch.setattr(ai_analysis, "get_vectorstore", lambda collection_name: FakeVectorStore())

    cv_info = {
        "name": "Nguyen Van A",
        "career_objective": "AI engineer",
        "skills": ["Python"],
        "experience": [],
        "education": [],
    }

    result = asyncio.run(ai_analysis.analyze_cv_against_jobs(cv_info, top_k=2))

    assert result["matched_jobs"][0]["job_title"] == "AI Engineer"
    assert result["matched_jobs"][0]["matched_skills"] == ["python"]
    assert any(item["skill"] == "machine learning" for item in result["learning_suggestions"])
    assert result["market_summary"]["top_jobs_found"] == 2
