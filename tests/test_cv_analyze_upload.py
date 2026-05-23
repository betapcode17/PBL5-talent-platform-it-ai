import io

from fastapi.testclient import TestClient

from app.main import app


class _FakeUploadFile:
    pass


def test_cv_analyze_upload_returns_insights_and_learning_suggestions(monkeypatch):
    import app.routers.cv as cv_router

    monkeypatch.setattr(cv_router, "extract_text_from_pdf", lambda path: "fake pdf text")
    monkeypatch.setattr(
        cv_router,
        "extract_cv_info",
        lambda text: {
            "name": "Nguyen Van A",
            "career_objective": "AI Engineer",
            "skills": ["Python", "ML"],
            "experience": [
                {
                    "title": "Data Analyst",
                    "company": "ABC",
                    "start_date": "2023-01-01",
                    "end_date": "Present",
                    "description": "Analyze data",
                }
            ],
            "education": [
                {
                    "school": "University",
                    "degree": "Bachelor",
                    "major": "Computer Science",
                    "start_date": "2019-09-01",
                    "end_date": "2023-06-30",
                }
            ],
        },
    )
    monkeypatch.setattr(cv_router, "insert_cv_record", lambda filename, cv_info, file_data: 321)

    async def fake_index_cv_extracts(skills, aspirations, experience, education, cv_id):
        return True

    async def fake_analyze_cv_insights(cv_info):
        return {
            "quality_score": 8.4,
            "completeness_score": 0.8,
            "has_portfolio": True,
            "has_certifications": False,
            "has_projects": True,
            "missing_sections": ["certifications"],
            "market_fit_score": 0.74,
            "experience_level": "Junior",
            "salary_range": "15-20 triệu",
            "competitive_score": 7.2,
            "strengths": ["Có nền tảng Python", "Có định hướng AI"],
            "weaknesses": ["Thiếu portfolio"],
        }

    async def fake_analyze_cv_against_jobs(cv_info, top_k=5):
        return {
            "matched_jobs": [
                {
                    "job_id": 11,
                    "job_title": "AI Engineer",
                    "company_name": "ACME AI",
                    "match_score": 0.91,
                    "matched_skills": ["python"],
                    "missing_skills": ["machine learning"],
                    "why_match": "Phu hop voi ky nang: python",
                    "salary": "20-30m",
                    "work_location": "HCM",
                    "work_type": "Full-time",
                }
            ],
            "learning_suggestions": [
                {
                    "skill": "machine learning",
                    "reason": "Xuat hien trong job phu hop",
                    "related_jobs_count": 3,
                    "example_jobs": ["AI Engineer", "ML Engineer"],
                    "priority": "high",
                }
            ],
            "market_summary": {"top_jobs_found": 1, "most_requested_skills": [{"skill": "machine learning", "count": 3}]},
        }

    monkeypatch.setattr(cv_router, "index_cv_extracts", fake_index_cv_extracts)
    monkeypatch.setattr(cv_router, "analyze_cv_insights", fake_analyze_cv_insights)
    monkeypatch.setattr(cv_router, "analyze_cv_against_jobs", fake_analyze_cv_against_jobs)
    monkeypatch.setattr(cv_router, "save_cv_insights", lambda cv_id, insights: None)

    client = TestClient(app)
    files = {"file": ("resume.pdf", io.BytesIO(b"%PDF-1.4 fake pdf"), "application/pdf")}
    response = client.post("/cv/analyze", files=files)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["cv_id"] == 321
    assert body["filename"] == "resume.pdf"
    assert body["insights"]["quality_score"] == 8.4
    assert body["insights"]["strengths"] == ["Có nền tảng Python", "Có định hướng AI"]
    assert body["matched_jobs"] == []
    assert body["learning_suggestions"][0]["skill"] == "machine learning"
    assert body["market_summary"]["top_jobs_found"] == 1
    # Debug: ensure extracted text is returned for inspection
    assert body.get("extracted_text") == "fake pdf text"


def test_cv_analyze_upload_allows_experience_only_cv(monkeypatch):
    import app.routers.cv as cv_router

    monkeypatch.setattr(cv_router, "extract_text_from_pdf", lambda path: "fake pdf text")
    monkeypatch.setattr(
        cv_router,
        "extract_cv_info",
        lambda text: {
            "name": "Nguyen Van B",
            "career_objective": "",
            "skills": [],
            "experience": [
                {
                    "title": "Backend Developer",
                    "company": "XYZ",
                    "start_date": "2022-01-01",
                    "end_date": "Present",
                    "description": "Build APIs with Python and FastAPI",
                }
            ],
            "education": [
                {
                    "school": "University",
                    "degree": "Bachelor",
                    "major": "Information Technology",
                    "start_date": "2018-09-01",
                    "end_date": "2022-06-30",
                }
            ],
        },
    )
    monkeypatch.setattr(cv_router, "insert_cv_record", lambda filename, cv_info, file_data: 322)

    async def fake_index_cv_extracts(skills, aspirations, experience, education, cv_id):
        return True

    async def fake_analyze_cv_insights(cv_info):
        return {
            "quality_score": 6.7,
            "completeness_score": 0.6,
            "has_portfolio": False,
            "has_certifications": False,
            "has_projects": False,
            "missing_sections": ["portfolio"],
            "market_fit_score": 0.52,
            "experience_level": "Mid",
            "salary_range": "20-30 triệu",
            "competitive_score": 6.1,
            "strengths": ["Có kinh nghiệm backend"],
            "weaknesses": ["Thiếu career objective"],
        }

    async def fake_analyze_cv_against_jobs(cv_info, top_k=5):
        return {
            "matched_jobs": [],
            "learning_suggestions": [],
            "market_summary": {"top_jobs_found": 0},
        }

    monkeypatch.setattr(cv_router, "index_cv_extracts", fake_index_cv_extracts)
    monkeypatch.setattr(cv_router, "analyze_cv_insights", fake_analyze_cv_insights)
    monkeypatch.setattr(cv_router, "analyze_cv_against_jobs", fake_analyze_cv_against_jobs)
    monkeypatch.setattr(cv_router, "save_cv_insights", lambda cv_id, insights: None)

    client = TestClient(app)
    files = {"file": ("backend-cv.pdf", io.BytesIO(b"%PDF-1.4 fake pdf"), "application/pdf")}
    response = client.post("/cv/analyze", files=files)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["cv_id"] == 322
    assert body["filename"] == "backend-cv.pdf"
    assert body["insights"]["strengths"] == ["Có kinh nghiệm backend"]


def test_cv_analyze_upload_falls_back_when_extraction_is_empty(monkeypatch):
    import app.routers.cv as cv_router

    monkeypatch.setattr(cv_router, "extract_text_from_pdf", lambda path: "OCR text with no structured sections")
    monkeypatch.setattr(
        cv_router,
        "extract_cv_info",
        lambda text: {
            "name": "",
            "career_objective": "",
            "skills": [],
            "experience": [],
            "education": [],
        },
    )
    monkeypatch.setattr(cv_router, "insert_cv_record", lambda filename, cv_info, file_data: 323)

    async def fake_index_cv_extracts(skills, aspirations, experience, education, cv_id):
        return True

    monkeypatch.setattr(cv_router, "index_cv_extracts", fake_index_cv_extracts)

    client = TestClient(app)
    files = {"file": ("scanned.pdf", io.BytesIO(b"%PDF-1.4 fake pdf"), "application/pdf")}
    response = client.post("/cv/analyze", files=files)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["cv_id"] == 323
    assert body["filename"] == "scanned.pdf"
    assert body["insights"]["weaknesses"]
    assert body["market_summary"]["top_jobs_found"] == 0
