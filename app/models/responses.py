from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from .core import Education, Experience, MatchedJob, Suggestion, ModelName, DeleteFileRequest, DocumentInfo, DocumentListResponse, MatchInput, JobDetails

class MatchResponse(BaseModel):
    name: Optional[str] = Field(None, description="Tên ứng viên")
    email: Optional[str] = Field(None, description="Email ứng viên")
    phone: Optional[str] = Field(None, description="Số điện thoại ứng viên")
    cv_skills: List[str] = Field(default_factory=list, description="Kỹ năng từ CV")
    career_objective: Optional[str] = Field(None, description="Mục tiêu nghề nghiệp")
    education: List[Education] = Field(default_factory=list, description="Học vấn")
    experience: List[Experience] = Field(default_factory=list, description="Kinh nghiệm")
    matched_jobs: List[MatchedJob] = Field(default_factory=list, description="Danh sách công việc khớp")
    suggestions: List[Suggestion] = Field(default_factory=list, description="Gợi ý cải thiện")
    session_id: Optional[str] = Field(None, description="ID phiên khớp")
    model: ModelName = Field(..., description="Mô hình AI sử dụng")


class CVInsightsResponse(BaseModel):
    """Response cho endpoint /cv/{cv_id}/insights"""
    cv_id: int = Field(..., description="ID của CV")
    quality_score: float = Field(..., ge=0.0, le=10.0, description="Điểm chất lượng CV (0-10)")
    completeness: Dict[str, Any] = Field(..., description="Độ đầy đủ của CV")
    market_fit: Dict[str, Any] = Field(..., description="Mức độ phù hợp với thị trường")
    strengths: List[str] = Field(default_factory=list, description="Điểm mạnh")
    weaknesses: List[str] = Field(default_factory=list, description="Điểm yếu")
    last_analyzed: Optional[str] = Field(None, description="Thời gian phân tích gần nhất")

class ImprovementSuggestion(BaseModel):
    """Gợi ý cải thiện cho từng section"""
    section: str = Field(..., description="Section cần cải thiện (skills, experience, education...)")
    current: Optional[List[str]] = Field(None, description="Giá trị hiện tại")
    suggested_add: Optional[List[str]] = Field(None, description="Gợi ý thêm vào")
    suggestion: str = Field(..., description="Mô tả gợi ý")
    reason: str = Field(..., description="Lý do")
    priority: str = Field(..., description="Mức độ ưu tiên (high, medium, low)")
    impact: str = Field(..., description="Tác động dự kiến")

class CVImproveResponse(BaseModel):
    """Response cho endpoint /cv/improve"""
    cv_id: int = Field(..., description="ID của CV")
    improvements: List[ImprovementSuggestion] = Field(..., description="Danh sách gợi ý cải thiện")

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

class DocumentPreviewResponse(BaseModel):
    """Response cho endpoint /preview-doc/{file_id}"""
    file_id: int = Field(..., description="ID file")
    type: str = Field(..., description="Loại file (cv, cover_letter...)")
    filename: str = Field(..., description="Tên file")
    preview: Dict[str, Any] = Field(..., description="Thông tin preview")
    quick_info: Dict[str, Any] = Field(..., description="Thông tin nhanh")

class SuggestQuestionsInput(BaseModel):
    """Input cho endpoint /suggest-questions"""
    context: str = Field(..., description="Context (cv_uploaded, viewing_job, chatting...)")
    cv_id: Optional[int] = Field(None, description="ID CV")
    job_id: Optional[int] = Field(None, description="ID job")

class QuestionSuggestion(BaseModel):
    """Gợi ý câu hỏi"""
    question: str = Field(..., description="Câu hỏi")
    category: str = Field(..., description="Danh mục (cv_analysis, improvement, salary...)")
    icon: str = Field(..., description="Icon emoji")

class SuggestQuestionsResponse(BaseModel):
    """Response cho endpoint /suggest-questions"""
    suggestions: List[QuestionSuggestion] = Field(..., description="Danh sách câu hỏi gợi ý")



class MatchExplanationPart(BaseModel):
    matched: List[str] = Field(default_factory=list)
    missing: List[str] = Field(default_factory=list)


class MatchExplanation(BaseModel):
    skills: Union[MatchExplanationPart, str, dict] = Field(default_factory=dict)
    experience: Union[MatchExplanationPart, str, dict] = ""
    education: Union[MatchExplanationPart, str, dict] = ""
    aspirations: Union[MatchExplanationPart, str, dict] = ""


class MatchedJob(BaseModel):
    job_id: str
    job_title: str
    job_url: str

    work_location: str
    salary: str
    deadline: str
    benefits: str
    job_type: str
    experience_required: str
    education_required: str
    company_name: str
    skills: List[str]

    match_score: float

    explanation: MatchExplanation
    why_match: str

    job_description: str