"""
Chart insights prompts for job analytics.
Dict of ChatPromptTemplate keyed by chart_type.
Used in routers.jobs.generate_chart_insights().
"""

from langchain_core.prompts import ChatPromptTemplate

chart_insights_prompts = {
    "top_jobs": ChatPromptTemplate.from_messages([
        (
            "human",
            """Bạn là chuyên gia phân tích thị trường việc làm. Phân tích biểu đồ "Top 10 Vị Trí Tuyển Dụng Nhiều Nhất" dựa trên dữ liệu sau:
{data_str}
Cung cấp phân tích ngắn gọn dưới dạng văn bản liên tục (3-4 câu), không sử dụng tiêu đề chào hỏi, không đánh số hoặc bullet points, chỉ tập trung vào nội dung chính:
- Xu hướng tuyển dụng chính
- Ngành nghề hot nhất
- Cơ hội cho ứng viên
- Lời khuyên
Trả lời bằng tiếng Việt, chuyên nghiệp, súc tích."""
        )
    ]),
    "top_companies": ChatPromptTemplate.from_messages([
        (
            "human",
            """Bạn là chuyên gia phân tích thị trường việc làm. Phân tích biểu đồ "Top 10 Công Ty Tuyển Dụng Nhiều Nhất" dựa trên dữ liệu sau:
{data_str}
Cung cấp phân tích ngắn gọn dưới dạng văn bản liên tục (3-4 câu), không sử dụng tiêu đề chào hỏi, không đánh số hoặc bullet points, chỉ tập trung vào nội dung chính:
- Công ty đang mở rộng
- Lĩnh vực kinh doanh
- Cơ hội phát triển
- Lời khuyên cho ứng viên
Trả lời bằng tiếng Việt, chuyên nghiệp, súc tích."""
        )
    ]),
    "location": ChatPromptTemplate.from_messages([
        (
            "human",
            """Bạn là chuyên gia phân tích thị trường việc làm. Phân tích biểu đồ "Phân Bố Địa Điểm Làm Việc" dựa trên dữ liệu sau:
{data_str}
Cung cấp phân tích ngắn gọn dưới dạng văn bản liên tục (3-4 câu), không sử dụng tiêu đề chào hỏi, không đánh số hoặc bullet points, chỉ tập trung vào nội dung chính:
- Thành phố có nhiều cơ hội nhất
- Xu hướng phân bố địa lý
- So sánh các thành phố
- Lời khuyên theo địa điểm
Trả lời bằng tiếng Việt, chuyên nghiệp, súc tích."""
        )
    ]),
    "job_type": ChatPromptTemplate.from_messages([
        (
            "human",
            """Bạn là chuyên gia phân tích thị trường việc làm. Phân tích biểu đồ "Phân Bố Loại Hình Công Việc" dựa trên dữ liệu sau:
{data_str}
Cung cấp phân tích ngắn gọn dưới dạng văn bản liên tục (3-4 câu), không sử dụng tiêu đề chào hỏi, không đánh số hoặc bullet points, chỉ tập trung vào nội dung chính:
- Loại hình phổ biến nhất
- Xu hướng remote/hybrid/onsite
- Cơ hội freelance/part-time
- Lời khuyên theo loại hình
Trả lời bằng tiếng Việt, chuyên nghiệp, súc tích."""
        )
    ]),
    "experience": ChatPromptTemplate.from_messages([
        (
            "human",
            """Bạn là chuyên gia phân tích thị trường việc làm. Phân tích biểu đồ "Phân Bố Yêu Cầu Kinh Nghiệm" dựa trên dữ liệu sau:
{data_str}
Cung cấp phân tích ngắn gọn dưới dạng văn bản liên tục (3-4 câu), không sử dụng tiêu đề chào hỏi, không đánh số hoặc bullet points, chỉ tập trung vào nội dung chính:
- Mức kinh nghiệm được yêu cầu nhiều
- Cơ hội cho fresher vs experienced
- Xu hướng tuyển dụng
- Lời khuyên cho từng nhóm
Trả lời bằng tiếng Việt, chuyên nghiệp, súc tích."""
        )
    ]),
    "salary": ChatPromptTemplate.from_messages([
        (
            "human",
            """Bạn là chuyên gia phân tích thị trường việc làm. Phân tích biểu đồ "Phân Bố Mức Lương" dựa trên dữ liệu sau:
{data_str}
Cung cấp phân tích ngắn gọn dưới dạng văn bản liên tục (3-4 câu), không sử dụng tiêu đề chào hỏi, không đánh số hoặc bullet points, chỉ tập trung vào nội dung chính:
- Mức lương phổ biến
- Xu hướng lương theo ngành
- So sánh thị trường
- Lời khuyên thương lượng lương
Trả lời bằng tiếng Việt, chuyên nghiệp, súc tích."""
        )
    ])
}