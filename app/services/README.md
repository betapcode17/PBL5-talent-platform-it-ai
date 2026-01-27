# 🧠 SERVICES - Business Logic

**Vị trị:** `e:\HOCKI6\PBL5\AI\app\services\`

## Mục đích

Xử lý business logic, không phụ thuộc vào HTTP framework

- LLM API calls
- Database operations
- Matching algorithms
- RAG pipeline
- Vector embeddings

## Các file

### ai_analysis.py

LLM-based analysis:

- `analyze_cv()` - Phân tích CV content
- `improve_cv()` - Gợi ý cải thiện
- `answer_question()` - Q&A
- `generate_insights()` - Tạo charts/insights

**Công nghệ:** OpenAI API, custom prompts

### candidate_matching.py

Matching algorithm v1:

- `match_cv_to_job()` - Tìm CV phù hợp cho Job
- `calculate_score()` - Tính similarity score
- `filter_candidates()` - Filter theo tiêu chí

**Công nghệ:** Semantic similarity, cosine distance

### rag_matching.py

RAG (Retrieval Augmented Generation):

- `rag_search()` - Tìm kiếm with RAG
- `rerank_candidates()` - Xếp hạng lại
- `generate_explanation()` - Giải thích tại sao match

**Công nghệ:** Vector DB (Chroma), LLM

### chroma_utils.py

Vector database utilities:

- `create_embeddings()` - Tạo embeddings
- `store_vectors()` - Lưu vào Chroma
- `search_vectors()` - Tìm kiếm vector
- `delete_vectors()` - Xóa vectors

**Công nghệ:** Chroma, Sentence-BERT

### db_utils.py

Database operations:

- `get_cv_from_db()` - Query CV
- `save_cv_to_db()` - Save CV
- `update_cv_in_db()` - Update CV
- `get_job_from_db()` - Query Job

**Công nghệ:** SQLite, Pandas

### api_key_manager.py

API key management:

- `get_api_key()` - Lấy API key
- `validate_key()` - Validate
- `refresh_key()` - Refresh (nếu cần)

### match_explain.py

Giải thích kết quả matching:

- `explain_match()` - Giải thích chi tiết
- `calculate_score_breakdown()` - Breakdown điểm
- `generate_report()` - Tạo report

### rag_helpers.py

Helper functions cho RAG:

- `chunk_text()` - Chia text thành chunks
- `create_prompts()` - Tạo prompts
- `format_context()` - Format context

## Cách sử dụng

```python
# routers/cv.py
from app.services import ai_analysis, candidate_matching

@app.post("/cv/analyze")
def analyze_cv(cv_id: str):
    # Call service function
    result = ai_analysis.analyze_cv(cv_id)
    return result

@app.post("/matching/search")
def find_matches(job_id: str):
    # Call matching service
    matches = candidate_matching.match_cv_to_job(job_id)
    return matches
```

## Best practices

✓ Services không import routers  
✓ Services không return HTTP responses  
✓ Services nhận & return Python objects/dicts  
✓ Error handling bằng exceptions (để routers xử lý)  
✓ Logging để debug  
✓ Caching nếu cần performance

```python
# Good example
def match_cv_to_job(job_id: str) -> List[dict]:
    """Match CVs to job, return list of matches"""
    job = db_utils.get_job(job_id)
    if not job:
        raise ValueError(f"Job {job_id} not found")

    matches = []
    for cv in db_utils.get_all_cvs():
        score = calculate_similarity(job, cv)
        if score > 0.4:
            matches.append({
                "cv_id": cv["id"],
                "score": score,
                "rank": len(matches) + 1
            })

    return sorted(matches, key=lambda x: x["score"], reverse=True)
```

```python
# Router xử lý response
@app.post("/matching/search")
def search(job_id: str):
    try:
        matches = services.match_cv_to_job(job_id)
        return SuccessResponse(data=matches)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
```

---

**Xem thêm:** `2_Tai_Lieu_Ky_Thuat/ADVANCED_MATCHING_V2.md`
