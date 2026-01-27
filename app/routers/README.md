# 🛣️ ROUTERS - API Endpoints

**Vị trí:** `e:\HOCKI6\PBL5\AI\app\routers\`

## Mục đích

Định nghĩa các API endpoints, route handling, request/response

## Các file

### cv.py - CV Management

**Endpoints:**

- `POST /cv/upload` - Upload CV file
- `GET /cv/{id}` - Get CV info
- `GET /cv/` - List all CVs
- `PUT /cv/{id}` - Update CV
- `DELETE /cv/{id}` - Delete CV
- `POST /cv/analyze` - Analyze with LLM
- `POST /cv/improve` - Get improvement suggestions

### jobs.py - Job Management

**Endpoints:**

- `GET /jobs/` - List jobs (with pagination)
- `GET /jobs/{id}` - Get job details
- `POST /jobs/` - Create new job
- `PUT /jobs/{id}` - Update job
- `DELETE /jobs/{id}` - Delete job
- `GET /jobs/search` - Search jobs by keyword

### matching.py - CV-Job Matching

**Endpoints:**

- `POST /matching/search` - Find CVs for a job
- `POST /matching/explain` - Explain why CV matches
- `GET /matching/stats` - Get matching statistics

### candidates.py - Candidate Management

**Endpoints:**

- `GET /candidates/` - List all candidates
- `GET /candidates/{id}` - Get candidate details
- `POST /candidates/filter` - Filter by criteria
- `GET /candidates/export` - Export candidates

### utils.py

Helper functions dùng chung cho routers

## Cách tạo endpoint mới

```python
# routers/new_feature.py
from fastapi import APIRouter, HTTPException, Depends
from app.models.core import SomeModel
from app.services import some_service

router = APIRouter(prefix="/api/v1/feature", tags=["Feature"])

@router.get("/")
def list_features():
    """List all features"""
    return some_service.get_all()

@router.get("/{id}")
def get_feature(id: str):
    """Get feature by ID"""
    result = some_service.get_by_id(id)
    if not result:
        raise HTTPException(status_code=404, detail="Not found")
    return result

@router.post("/")
def create_feature(data: SomeModel):
    """Create new feature"""
    return some_service.create(data)
```

## Include router trong main.py

```python
# app/main.py
from fastapi import FastAPI
from app.routers import cv, jobs, matching, candidates

app = FastAPI(title="CV Screener API")

# Include routers
app.include_router(cv.router, tags=["CV"])
app.include_router(jobs.router, tags=["Jobs"])
app.include_router(matching.router, tags=["Matching"])
app.include_router(candidates.router, tags=["Candidates"])
```

## Best practices

✓ Tách router theo tính năng  
✓ Dùng dependency injection (`Depends()`)  
✓ Validate request data qua models  
✓ Trả về response models cụ thể  
✓ Xử lý errors với `HTTPException`  
✓ Thêm documentation (docstrings)

## Testing endpoints

```bash
# Tất cả endpoints tự động documented tại:
http://localhost:8000/docs

# Hoặc tài liệu ReDoc:
http://localhost:8000/redoc
```

---

**Xem thêm:** `2_Tai_Lieu_Ky_Thuat/API_EXAMPLES.md`
