# 📦 MODELS - Data Structures

**Vị trí:** `e:\HOCKI6\PBL5\AI\app\models\`

## Mục đích

Định nghĩa các Pydantic models cho:

- Request validation
- Response serialization
- Type hints & auto-documentation

## Các file

### core.py

Định nghĩa các data models chính:

- `CVModel` - Thông tin CV
- `JobModel` - Thông tin Job
- `CandidateModel` - Candidate info
- `MatchResult` - Kết quả matching

### responses.py

API response schemas:

- `SuccessResponse` - Response thành công
- `ErrorResponse` - Error message
- `PaginatedResponse` - Paginated results
- `MatchResultResponse` - Matching results

## Cách sử dụng

```python
from app.models.core import CVModel, JobModel
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Request validation tự động
@app.post("/cv/")
def create_cv(cv: CVModel):
    # cv đã được validate theo schema
    return {"status": "created"}

# Response serialization tự động
@app.get("/cv/{id}", response_model=CVModel)
def get_cv(id: str):
    # Return dict hoặc model, fastapi sẽ serialize thành JSON
    return {...}
```

## Best practices

✓ Mỗi model inherit từ `BaseModel`  
✓ Thêm type hints đầy đủ  
✓ Dùng `Optional[type]` cho fields không bắt buộc  
✓ Thêm doc strings cho mỗi model  
✓ Validate data bằng `Field()` validators

```python
from pydantic import BaseModel, Field, EmailStr

class CVModel(BaseModel):
    id: str = Field(..., description="Unique CV ID")
    email: EmailStr = Field(..., description="Email address")
    name: str = Field(..., min_length=1, max_length=100)
    experience_years: int = Field(..., ge=0, le=70)

    class Config:
        schema_extra = {
            "example": {
                "id": "CV001",
                "email": "john@example.com",
                "name": "John Doe",
                "experience_years": 5
            }
        }
```

---

**Xem thêm:** `2_Tai_Lieu_Ky_Thuat/API_EXAMPLES.md`
