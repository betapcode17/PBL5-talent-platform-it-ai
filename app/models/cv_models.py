"""
CV-related Pydantic models (Education, Experience).
"""

from pydantic import BaseModel, Field, field_validator
from typing import List
from datetime import datetime
import re

class Education(BaseModel):
    school: str = Field(..., min_length=1, description="Tên trường học")
    degree: str = Field(..., min_length=1, description="Bằng cấp")
    major: str = Field(..., min_length=1, description="Chuyên ngành")
    start_date: str = Field(..., description="Ngày bắt đầu (YYYY-MM-DD)")
    end_date: str = Field(..., description="Ngày kết thúc (YYYY-MM-DD hoặc 'Present')")

    @field_validator("start_date")
    @classmethod
    def validate_start_date(cls, value):
        """Kiểm tra định dạng ngày YYYY-MM-DD."""
        if not re.match(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError("Date must be in YYYY-MM-DD format")
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Invalid date")
        return value

    @field_validator("end_date")
    @classmethod
    def validate_end_date(cls, value):
        """Kiểm tra định dạng ngày YYYY-MM-DD hoặc 'Present'."""
        if value != "Present" and not re.match(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError("Date must be in YYYY-MM-DD format or 'Present'")
        if value != "Present":
            try:
                datetime.strptime(value, "%Y-%m-%d")
            except ValueError:
                raise ValueError("Invalid date")
        return value

class Experience(BaseModel):
    company: str = Field(..., min_length=1, description="Tên công ty")
    title: str = Field(..., min_length=1, description="Chức danh")
    start_date: str = Field(..., description="Ngày bắt đầu (YYYY-MM-DD)")
    end_date: str = Field(..., description="Ngày kết thúc (YYYY-MM-DD hoặc 'Present')")
    description: str = Field(..., min_length=1, description="Mô tả công việc")

    @field_validator("company", mode="before")
    @classmethod
    def ensure_non_empty_company(cls, value):
        """Đảm bảo trường company không rỗng, gán 'Unknown' nếu rỗng hoặc None."""
        return value or "Unknown"

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_date_format(cls, value):
        """Kiểm tra định dạng ngày YYYY-MM-DD hoặc 'Present'."""
        if value != "Present" and not re.match(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError("Date must be in YYYY-MM-DD format or 'Present'")
        if value != "Present":
            try:
                datetime.strptime(value, "%Y-%m-%d")
            except ValueError:
                raise ValueError("Invalid date")
        return value