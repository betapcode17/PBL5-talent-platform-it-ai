"""
Prompt for CV improvement suggestions.
Used in ai_analysis.generate_cv_improvements().
"""

from langchain_core.prompts import ChatPromptTemplate

cv_improvement_prompt = ChatPromptTemplate.from_messages([
    (
        "human",
        """Bạn là chuyên gia tư vấn CV. Dựa trên phân tích CV, hãy đưa ra gợi ý cải thiện CỤ THỂ DỰA TRÊN CV HIỆN TẠI:

**THÔNG TIN CV HIỆN TẠI:**
- Kỹ năng: {skills}
- Số lượng kinh nghiệm: {experience_count} công việc
- Số lượng học vấn: {education_count} bằng cấp
- Mục tiêu nghề nghiệp: {career_objective}
- Điểm yếu đã phát hiện: {weaknesses}
- Phần thiếu: {missing_sections}
- Có Portfolio: {has_portfolio}
- Có Certifications: {has_certifications}
- Có Projects: {has_projects}

**YÊU CẦU GỢI Ý:**
Đưa ra 5 gợi ý cải thiện CỤ THỂ, DỰA TRÊN CV HIỆN TẠI:

1. **Nếu thiếu kỹ năng:** Gợi ý thêm kỹ năng cụ thể (dựa trên ngành nghề)
2. **Nếu thiếu dự án:** Gợi ý thêm mục Projects với ví dụ cụ thể
3. **Nếu mô tả kinh nghiệm chung chung:** Gợi ý thêm metrics, con số cụ thể
4. **Nếu mục tiêu nghề nghiệp mơ hồ:** Gợi ý làm rõ hơn
5. **Nếu thiếu portfolio/certifications:** Gợi ý thêm

**ĐỊNH DẠNG JSON:**
[
  {{
    "section": "skills",
    "current": ["Python", "JavaScript"],
    "suggested_add": ["React", "Node.js"],
    "suggestion": "Thêm kỹ năng React và Node.js để tăng cơ hội với vị trí Full-stack Developer",
    "reason": "70% jobs Full-stack yêu cầu React, Node.js đang rất hot trong thị trường",
    "priority": "high",
    "impact": "+40% match rate với jobs Full-stack"
  }},
  {{
    "section": "projects",
    "current": null,
    "suggested_add": null,
    "suggestion": "Thêm mục 'Dự án cá nhân' với 2-3 projects showcase kỹ năng Python, JavaScript",
    "reason": "Projects giúp nhà tuyển dụng thấy được năng lực thực tế, tăng độ tin cậy",
    "priority": "high",
    "impact": "Tăng 50% cơ hội được phỏng vấn"
  }},
  {{
    "section": "experience",
    "current": null,
    "suggested_add": null,
    "suggestion": "Thêm metrics vào mô tả công việc (VD: 'Tối ưu API giảm 30% response time', 'Xây dựng feature tăng 20% user engagement')",
    "reason": "Metrics cụ thể giúp CV nổi bật hơn, chứng minh impact thực tế",
    "priority": "medium",
    "impact": "Tăng độ tin cậy và chuyên nghiệp"
  }},
  {{
    "section": "career_objective",
    "current": null,
    "suggested_add": null,
    "suggestion": "Làm rõ mục tiêu nghề nghiệp: Thay vì 'Tìm cơ hội phát triển', hãy viết 'Mong muốn trở thành Full-stack Developer tại công ty công nghệ, đóng góp vào các dự án web app quy mô lớn'",
    "reason": "Mục tiêu cụ thể giúp nhà tuyển dụng hiểu rõ định hướng của bạn",
    "priority": "medium",
    "impact": "Tăng sự phù hợp với job description"
  }},
  {{
    "section": "certifications",
    "current": null,
    "suggested_add": null,
    "suggestion": "Thêm chứng chỉ liên quan (VD: AWS Certified Developer, Google Cloud Associate)",
    "reason": "Certifications chứng minh năng lực và sự đầu tư vào nghề nghiệp",
    "priority": "low",
    "impact": "+15% match rate với jobs yêu cầu cloud"
  }}
]

**LƯU Ý:**
- Gợi ý phải CỤ THỂ, DỰA TRÊN CV HIỆN TẠI
- Không gợi ý chung chung như "Cải thiện CV"
- Phải có lý do và impact rõ ràng

Chỉ trả về JSON array, không giải thích thêm."""
    )
])