"""
Job search Pydantic models.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class JobSearchInput(BaseModel):
    """Input cho endpoint /jobs/search"""
    query: Optional[str] = Field(None, description="Từ khóa tìm kiếm")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Bộ lọc")
    cv_id: Optional[int] = Field(None, description="ID CV để AI ranking")
    limit: int = Field(20, ge=1, le=100, description="Số lượng kết quả")
    offset: int = Field(0, ge=0, description="Offset cho pagination")

class JobSearchResult(BaseModel):
    """Kết quả tìm kiếm job"""
    job_id: int = Field(..., description="ID job")
    job_title: str = Field(..., description="Tiêu đề job")
    company_name: str = Field(..., description="Tên công ty")
    match_score: Optional[float] = Field(None, description="Điểm match (nếu có cv_id)")
    salary: str = Field(..., description="Mức lương")
    work_location: str = Field(..., description="Địa điểm")
    work_type: str = Field(..., description="Loại hình")
    deadline: str = Field(..., description="Hạn nộp")
    why_match: Optional[str] = Field(None, description="Lý do phù hợp")

    class Config:
        populate_by_name = True  # Allow both field name and alias

class JobSearchResponse(BaseModel):
    """Response cho endpoint /jobs/search"""
    total: int = Field(..., description="Tổng số jobs tìm được")
    jobs: List[JobSearchResult] = Field(..., description="Danh sách jobs")
    limit: int = Field(..., description="Limit")
    offset: int = Field(..., description="Offset")