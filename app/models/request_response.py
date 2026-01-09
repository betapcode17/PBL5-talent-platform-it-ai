"""
Request and Response Pydantic models.
"""

from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import List, Dict, Any, Optional
from datetime import datetime
import re

class Suggestion(BaseModel):
    skill_or_experience: str = Field(..., min_length=1, description="Kỹ năng hoặc kinh nghiệm cần cải thiện")
    suggestion: str = Field(..., min_length=1, description="Gợi ý cải thiện")

class DocumentInfo(BaseModel):
    id: int = Field(..., ge=1, description="ID của CV trong cv_store")
    filename: str = Field(..., min_length=1, description="Tên file CV")
    cv_info_json: str = Field(..., min_length=1, description="JSON thông tin CV")
    upload_timestamp: Optional[str] = Field(None, description="Thời gian tạo (timestamp)")

class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo] = Field(..., description="Danh sách thông tin CV")

class DeleteFileRequest(BaseModel):
    file_id: int = Field(..., ge=1, description="ID của CV cần xóa")

class MatchInput(BaseModel):
    model_config = ConfigDict(
        schema_extra={
            "example": {
                "session_id": "test-session-123",
                "model": "gemini-2.5-flash",
                "cv_input": "Skills: Python, SQL\nAspirations: Senior Data Scientist\nExperience: Company: Tech Corp; Title: Data Analyst; Start_date: 2020-01-01; End_date: Present; Description: Analyzed data\nEducation: School: Hanoi University; Degree: Bachelor; Major: Computer Science; Start_date: 2016-09-01; End_date: 2020-06-30\nName: John Doe\nEmail: john.doe@example.com\nPhone: +84-123-456-789",
                "cv_id": 1,
                "filters": {
                    "job_type": ["Full-time"],
                    "work_location": ["Hà Nội"],
                    "skills": ["Python", "SQL"]
                }
            }
        }
    )

    session_id: Optional[str] = Field(None, description="ID phiên khớp, nếu có")
    model: ModelName = Field(ModelName.gemini_2_5_flash, description="Mô hình AI sử dụng")
    cv_input: Optional[str] = Field(None, description="Input CV dạng chuỗi")
    cv_id: Optional[int] = Field(None, ge=1, description="ID của CV trong cv_store")
    filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Bộ lọc công việc. Allowed keys: 'job_type', 'work_location', 'experience', 'education', 'skills', 'deadline_after'. Example: {'job_type': ['Full-time'], 'work_location': ['Hà Nội'], 'skills': ['Python']}"
    )

    @field_validator("filters")
    @classmethod
    def validate_filters(cls, value):
        """Kiểm tra các khóa và giá trị hợp lệ trong filters."""
        valid_keys = {"job_type", "work_location", "experience", "education", "skills", "deadline_after"}
        for key in value:
            if key not in valid_keys:
                raise ValueError(
                    f"Invalid filter key: '{key}'. Allowed keys are: {', '.join(valid_keys)}."
                )
            if key in {"job_type", "skills"} and not isinstance(value[key], list):
                raise ValueError(f"Filter '{key}' must be a list")
            if key == "deadline_after" and not re.match(r"\d{4}-\d{2}-\d{2}", value[key]):
                raise ValueError("deadline_after must be in YYYY-MM-DD format")
        return value

    @field_validator("cv_input", "cv_id", mode="before")
    @classmethod
    def ensure_cv_input_or_id(cls, value, info):
        """Đảm bảo ít nhất một trong cv_input hoặc cv_id được cung cấp."""
        values = info.data
        if info.field_name == "cv_id" and value is None and values.get("cv_input") is None:
            raise ValueError("At least one of cv_input or cv_id must be provided")
        return value