"""
Prompt for CV insights analysis.
Used in ai_analysis.analyze_cv_insights().
"""

from langchain_core.prompts import ChatPromptTemplate

cv_analysis_prompt = ChatPromptTemplate.from_messages([
    (
        "human",
        """Bạn là chuyên gia phân tích CV. Hãy phân tích CV sau và đưa ra đánh giá chi tiết:

**THÔNG TIN CV:**
- Tên: {name}
- Email: {email}
- Điện thoại: {phone}
- Kỹ năng: {skills}
- Mục tiêu nghề nghiệp: {career_objective}
- Kinh nghiệm: {experience_count} công việc
- Học vấn: {education_count} bằng cấp

**YÊU CẦU PHÂN TÍCH:**

1. **Quality Score (0-10):** Đánh giá tổng thể chất lượng CV
2. **Completeness Score (0-1):** Độ đầy đủ của CV
   - Tính theo công thức: (Số phần có / Tổng số phần cần thiết)
   - Các phần cần thiết: Name, Email, Phone, Skills, Career Objective, Experience, Education
   - Các phần bổ sung: Portfolio, Certifications, Projects, Awards
   - **LƯU Ý:** Điểm này KHÔNG BAO GIỜ ÂM, tối thiểu là 0.0
   - Ví dụ: Có 7/10 phần → completeness_score = 0.7
3. **Điểm mạnh:** 3-5 điểm dựa trên CV hiện tại
4. **Điểm yếu:** 3-5 điểm dựa trên những gì CV thiếu hoặc cần cải thiện
5. **Market Fit Score (0-1):** Mức độ phù hợp với thị trường việc làm

**ĐỊNH DẠNG JSON RESPONSE:**
{{
  "quality_score": 7.5,
  "completeness_score": 0.7,
  "has_portfolio": false,
  "has_certifications": false,
  "has_projects": false,
  "missing_sections": ["Portfolio", "Certifications", "Projects"],
  "market_fit_score": 0.65,
  "experience_level": "Junior",
  "salary_range": "8-12 triệu",
  "competitive_score": 6.8,
  "strengths": [
    "Có kỹ năng Adobe Photoshop tốt",
    "Có kinh nghiệm thực tế với Social Media",
    "Mục tiêu nghề nghiệp rõ ràng"
  ],
  "weaknesses": [
    "Thiếu kỹ năng Illustrator (cần cho 80% jobs Marketing Designer)",
    "Chưa có portfolio online để showcase sản phẩm",
    "Kinh nghiệm chưa đủ 2 năm cho vị trí Senior"
  ]
}}

**LƯU Ý QUAN TRỌNG:**
- completeness_score PHẢI từ 0.0 đến 1.0, KHÔNG BAO GIỜ ÂM
- strengths và weaknesses phải cụ thể, dựa trên CV hiện tại
- missing_sections chỉ liệt kê những phần thực sự thiếu

Chỉ trả về JSON, không giải thích thêm."""
    )
])