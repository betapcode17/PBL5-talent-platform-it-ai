"""
Application Pydantic models.
"""

from pydantic import BaseModel, Field
from typing import List

class ApplyJobInput(BaseModel):
    """Input cho endpoint /apply"""
    cv_id: int = Field(..., ge=1, description="ID của CV")
    job_id: int = Field(..., ge=1, description="ID của job")
    cover_letter: str = Field("", description="Thư xin việc")
    status: str = Field("applied", description="Trạng thái (applied, pending...)")

class ApplicationResponse(BaseModel):
    """Response cho endpoint /apply"""
    application_id: int = Field(..., description="ID của application")
    cv_id: int = Field(..., description="ID CV")
    job_id: int = Field(..., description="ID job")
    status: str = Field(..., description="Trạng thái")
    applied_at: str = Field(..., description="Thời gian apply")
    message: str = Field(..., description="Thông báo")

class ApplicationItem(BaseModel):
    """Item trong danh sách applications"""
    id: int = Field(..., description="ID application")
    cv_id: int = Field(..., description="ID CV")
    job_id: int = Field(..., description="ID job")
    job_title: str = Field(..., description="Tiêu đề job")
    company_url: str = Field(..., description="URL công ty")
    salary: str = Field(..., description="Mức lương")
    work_location: str = Field(..., description="Địa điểm")
    status: str = Field(..., description="Trạng thái")
    applied_at: str = Field(..., description="Thời gian apply")

class ApplicationsResponse(BaseModel):
    """Response cho endpoint /applications/{cv_id}"""
    cv_id: int = Field(..., description="ID CV")
    total: int = Field(..., description="Tổng số applications")
    applications: List[ApplicationItem] = Field(..., description="Danh sách applications")