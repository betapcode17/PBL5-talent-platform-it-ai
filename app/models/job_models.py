"""
Job-related Pydantic models (MatchedJob, JobDetails).
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Optional
from datetime import datetime
import re

class MatchedJob(BaseModel):
    model_config = ConfigDict(populate_by_name=True)  # Allow alias

    job_id: int = Field(..., ge=1, description="ID công việc từ job_store")
    job_title: str = Field(..., min_length=1, description="Tiêu đề công việc")
    job_url: str = Field(..., description="URL công việc")
    match_score: float = Field(..., ge=0.0, le=1.0, description="Điểm khớp (0-1)")
    matched_skills: List[str] = Field(default_factory=list, description="Kỹ năng khớp với CV")
    matched_aspirations: List[str] = Field(default_factory=list, description="Mục tiêu nghề nghiệp khớp")
    matched_experience: List[str] = Field(default_factory=list, description="Kinh nghiệm khớp")
    matched_education: List[str] = Field(default_factory=list, description="Học vấn khớp")
    work_location: str = Field(..., min_length=1, description="Địa điểm làm việc")
    salary: str = Field(..., description="Mức lương")
    deadline: str = Field(..., description="Hạn nộp hồ sơ")
    benefits: str = Field(..., description="Phúc lợi")
    job_type: str = Field(..., min_length=1, description="Loại hình công việc")
    experience_required: str = Field(..., description="Kinh nghiệm yêu cầu")
    education_required: str = Field(..., description="Học vấn yêu cầu")
    company_name: str = Field(..., min_length=1, description="Tên công ty")
    skills: List[str] = Field(default_factory=list, description="Danh sách kỹ năng yêu cầu")
    why_match: Optional[str] = Field(None, description="Lý do tại sao phù hợp (AI-generated)")
    job_description: Optional[str] = Field(None, description="Mô tả công việc")

    @field_validator("deadline")
    @classmethod
    def validate_deadline(cls, value):
        """Kiểm tra định dạng ngày YYYY-MM-DD cho deadline."""
        if not re.match(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError("Deadline must be in YYYY-MM-DD format")
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Invalid deadline date")
        return value

class JobDetails(BaseModel):
    model_config = ConfigDict(populate_by_name=True)  # Allow alias for job_id = id

    job_id: int = Field(..., ge=1, description="ID công việc", alias="id")
    name: str = Field(..., min_length=1, description="Tên công việc")
    job_title: str = Field(..., min_length=1, description="Tiêu đề công việc")
    job_url: str = Field(..., description="URL công việc")
    job_description: str = Field(..., description="Mô tả công việc")
    candidate_requirements: str = Field(..., description="Yêu cầu ứng viên")
    benefits: str = Field(..., description="Phúc lợi")
    work_location: str = Field(..., min_length=1, description="Địa điểm làm việc")
    work_time: str = Field(..., description="Thời gian làm việc")
    job_tags: str = Field(..., description="Tag công việc")
    skills: List[str] = Field(default_factory=list, description="Kỹ năng yêu cầu")
    related_categories: str = Field(..., description="Danh mục liên quan")
    salary: str = Field(..., description="Mức lương")
    experience: str = Field(..., description="Kinh nghiệm yêu cầu")
    deadline: str = Field(..., description="Hạn nộp hồ sơ")
    company_logo: str = Field(..., description="Logo công ty")
    company_scale: str = Field(..., description="Quy mô công ty")
    company_field: str = Field(..., description="Lĩnh vực công ty")
    company_address: str = Field(..., description="Địa chỉ công ty")
    level: str = Field(..., description="Cấp bậc")
    education: str = Field(..., description="Học vấn yêu cầu")
    number_of_hires: int = Field(..., ge=0, description="Số lượng tuyển")
    work_type: str = Field(..., min_length=1, description="Loại hình công việc")
    company_url: str = Field(..., description="URL công ty")
    timestamp: str = Field(..., description="Thời gian tạo")

    @field_validator("skills", mode="before")
    @classmethod
    def parse_skills(cls, value):
        """Chuyển đổi chuỗi skills thành danh sách."""
        if isinstance(value, str):
            return [skill.strip() for skill in value.split(";") if skill.strip()]
        return value

    @field_validator("deadline")
    @classmethod
    def validate_deadline(cls, value):
        """Kiểm tra định dạng ngày YYYY-MM-DD cho deadline."""
        if not re.match(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError("Deadline must be in YYYY-MM-DD format")
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Invalid deadline date")
        return value