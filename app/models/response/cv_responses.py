"""
CV insights and improvement Pydantic models.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

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