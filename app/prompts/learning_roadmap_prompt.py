"""
Prompt for generating a concrete, time-boxed learning roadmap.
Used in ai_analysis.generate_learning_roadmap().
"""
from langchain_core.prompts import ChatPromptTemplate

learning_roadmap_prompt = ChatPromptTemplate.from_messages([
    (
        "human",
        """Bạn là cố vấn học tập nghề nghiệp. Dựa trên CV hiện tại và những kỹ năng còn thiếu/weaknesses, hãy tạo LỘ TRÌNH HỌC TẬP CỤ THỂ và CÓ THỂ THỰC HIỆN được.

Input:
- Kỹ năng hiện có: {skills}
- Những điểm yếu / thiếu: {weaknesses}
- Gợi ý học từ job data: {learning_suggestions}

Yêu cầu:
- Trả về JSON array gồm các bước (phases). Mỗi bước có: phase (mô tả, ví dụ '0-3 months'), duration_weeks (số tuần), objectives (danh sách mục tiêu ngắn), resources (danh sách object với 'name' và 'url'), projects (ví dụ project nhỏ để thực hành), milestones (cách đánh giá).
- Tối đa 4 bước: nhanh (0-4 tuần), cơ bản (1-3 months), nâng cao (3-6 months), chuyên sâu (6+ months) - nếu không cần thì bỏ bớt.
- Mỗi mục phải cụ thể, thực tế, có tên khóa học hoặc tài nguyên (ví dụ: 'Spring Boot Official Guide', 'Coursera: Java Programming: Solving Problems with Software').
- Ưu tiên kỹ năng theo priority (nếu learning_suggestions có priority)
- Chỉ trả về JSON, KHÔNG giải thích thêm.

Example output:
[
  {
    "phase": "0-4 weeks",
    "duration_weeks": 4,
    "objectives": ["Hiểu cú pháp Java cơ bản", "Thiết lập môi trường Spring Boot"],
    "resources": [{"name":"Java Programming - Coursera","url":"https://www.coursera.org/..."}],
    "projects": ["Build a RESTful API for notes app using Spring Boot"],
    "milestones": ["Complete 3 tutorials", "Deploy app to Heroku"]
  }
]

Chỉ trả về JSON array."""
    )
])
