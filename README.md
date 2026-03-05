<div align="center">

# 💬 AI Job Consulting Chatbot

### Chatbot RAG Tư vấn Nghề nghiệp & Tìm kiếm Việc làm

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Google Gemini](https://img.shields.io/badge/Gemini_2.5_Flash-8E75B2?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.17-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5.0-FF6F00?style=for-the-badge)](https://www.trychroma.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)](https://postgresql.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**Chatbot AI sử dụng RAG (Retrieval-Augmented Generation) để tư vấn nghề nghiệp, gợi ý việc làm, phân tích CV và hỗ trợ matching CV-Job — nền tảng trên Google Gemini, ChromaDB và PostgreSQL với hơn 6,000 việc làm thực tế.**

[Demo](#-demo-chatbot) •
[Bắt đầu nhanh](#-bắt-đầu-nhanh) •
[Cách hoạt động](#-cách-chatbot-hoạt-động) •
[API Endpoints](#-api-endpoints)

---

> **"Tìm cho tôi việc Python ở Đà Nẵng lương trên 20 triệu"**
> — Đó là tất cả những gì bạn cần nói. Chatbot sẽ tìm kiếm, phân tích và gợi ý việc làm phù hợp nhất.

</div>

---

## 📋 Mục lục

- [Tổng quan](#-tổng-quan)
- [Demo Chatbot](#-demo-chatbot)
- [Tính năng Chatbot](#-tính-năng-chatbot)
- [Cách Chatbot hoạt động](#-cách-chatbot-hoạt-động)
- [4 Chế độ tư vấn](#-4-chế-độ-tư-vấn)
- [Bắt đầu nhanh](#-bắt-đầu-nhanh)
- [API Endpoints](#-api-endpoints)
- [Kiến trúc RAG Pipeline](#-kiến-trúc-rag-pipeline)
- [Services chi tiết](#%EF%B8%8F-services-chi-tiết)
- [Data Models](#-data-models)
- [Database & Vector Store](#-database--vector-store)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Cấu hình](#%EF%B8%8F-cấu-hình)
- [Mở rộng: CV Screener](#-mở-rộng-cv-screener-module)

---

## 🎯 Tổng quan

**AI Job Consulting Chatbot** là chatbot tư vấn nghề nghiệp thông minh, được xây dựng cho đồ án PBL5. Thay vì tìm kiếm thủ công, người dùng chỉ cần **hỏi bằng ngôn ngữ tự nhiên** — chatbot sẽ hiểu ý định, tìm kiếm trong cơ sở dữ liệu 6,338 việc làm và trả lời bằng tiếng Việt:

```
👤 Người dùng:  "Tôi biết React và Node.js, có 2 năm kinh nghiệm, nên ứng tuyển gì?"

🤖 Chatbot:     Dựa trên kỹ năng React + Node.js và 2 năm kinh nghiệm, tôi gợi ý:

                📌 Frontend Developer — TechCorp
                   📍 Hà Nội | 💰 18-25 triệu | ✅ Match: React, Node.js

                📌 Fullstack Developer — FPT Software
                   📍 Đà Nẵng | 💰 20-30 triệu | ✅ Match: React, Node.js, JavaScript

                💡 Gợi ý: Bạn nên bổ sung TypeScript và Docker để mở rộng cơ hội...
```

### Công nghệ cốt lõi

|      Thành phần      |           Công nghệ           | Vai trò trong Chatbot                          |
| :------------------: | :---------------------------: | :--------------------------------------------- |
|      🧠 **LLM**      |    Google Gemini 2.5 Flash    | Hiểu câu hỏi, sinh câu trả lời thông minh      |
| 🔍 **Vector Search** | ChromaDB + text-embedding-004 | Tìm kiếm semantic đa ngôn ngữ trong 6,338 jobs |
| 🔗 **RAG Pipeline**  |       LangChain 0.1.17        | Kết nối retrieval → context → generation       |
|   🗄️ **Database**    |          PostgreSQL           | Lưu trữ & enrichment 6,338 jobs, 2,906 công ty |
|      ⚡ **API**      |            FastAPI            | Phục vụ chatbot qua REST API, WebSocket-ready  |

---

## 🎬 Demo Chatbot

### Các câu hỏi chatbot có thể trả lời

<table>
<tr>
<td width="50%">

**🔍 Tìm kiếm việc làm**

```
"Tìm việc Java ở Hồ Chí Minh"
"Có vị trí nào remote cho DevOps không?"
"Việc lương trên 30 triệu tại Đà Nẵng?"
"Senior Python Developer ở đâu đang tuyển?"
```

**📊 Phân tích thị trường**

```
"Ngôn ngữ lập trình nào hot nhất hiện nay?"
"Mức lương trung bình cho Frontend Developer?"
"So sánh Java và Python trên thị trường VN?"
"Xu hướng tuyển dụng IT 2026?"
```

</td>
<td width="50%">

**📄 Tư vấn CV**

```
"CV của tôi thiếu gì để ứng tuyển Backend?"
"Cách viết CV cho fresher IT?"
"Nên highlight kỹ năng nào trong CV?"
"CV tiếng Anh hay tiếng Việt tốt hơn?"
```

**🎯 Tư vấn nghề nghiệp**

```
"Tôi biết Python, nên học thêm gì?"
"Lộ trình từ Junior lên Senior Developer?"
"Fullstack hay chuyên sâu Backend?"
"Có nên chuyển từ Tester sang Dev không?"
```

</td>
</tr>
</table>

### Ví dụ request/response

```bash
curl -X POST http://localhost:8000/api/chat/message \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Có công việc Python nào ở Đà Nẵng không?",
    "context_type": "jobs"
  }'
```

```json
{
  "response": "Tôi tìm thấy một số vị trí Python tại Đà Nẵng cho bạn:\n\n📌 **Senior Python Developer** — FPT Software\n   📍 Đà Nẵng | 💰 25-35 triệu\n   🔧 Python, FastAPI, PostgreSQL, Docker\n\n📌 **Backend Engineer (Python)** — Sun Asterisk\n   📍 Đà Nẵng | 💰 20-28 triệu\n   🔧 Python, Django, AWS\n\n💡 Gợi ý: Nếu bạn có kinh nghiệm FastAPI hoặc Django, cơ hội sẽ rất cao...",
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "timestamp": "2026-03-05T10:30:00",
  "sources": [
    {
      "id": "42",
      "title": "Senior Python Developer",
      "company": "FPT Software",
      "location": "Đà Nẵng",
      "similarity": 0.92
    },
    {
      "id": "108",
      "title": "Backend Engineer (Python)",
      "company": "Sun Asterisk",
      "location": "Đà Nẵng",
      "similarity": 0.87
    }
  ],
  "confidence_score": 0.89
}
```

---

## ✨ Tính năng Chatbot

### 💬 Hội thoại tự nhiên bằng tiếng Việt

- Hiểu câu hỏi tiếng Việt tự nhiên, không cần từ khóa cứng
- Trả lời chi tiết, có cấu trúc, dễ đọc
- Gợi ý thêm thông tin liên quan (skills cần bổ sung, lộ trình phát triển...)
- Hỗ trợ hội thoại nhiều lượt (multi-turn) — chatbot nhớ ngữ cảnh

### 🔍 Tìm kiếm việc làm thông minh (RAG)

- **Semantic search**: Tìm theo ý nghĩa, không chỉ từ khóa (VD: "việc liên quan AI" → tìm cả Machine Learning, Data Science...)
- **6,338 việc làm** thực tế từ thị trường Việt Nam
- **2,906 công ty** với đầy đủ thông tin
- **6,371 kỹ năng** được phân loại
- Trả kèm **confidence score** & **nguồn tham khảo** cho mỗi câu trả lời

### 📄 Phân tích & Cải thiện CV

- Upload PDF → tự động extract (tên, kỹ năng, kinh nghiệm, học vấn)
- **Quality Score** (0–10): Đánh giá chất lượng tổng thể
- **Market Fit Score**: Mức phù hợp với thị trường việc làm hiện tại
- Gợi ý cải thiện chi tiết từng mục, kèm mức độ ưu tiên & impact

### 🔗 Matching CV ↔ Job

- Match CV với database việc làm qua RAG pipeline
- Giải thích lý do matching bằng tiếng Việt (skills khớp, kinh nghiệm, học vấn)
- Gợi ý kỹ năng cần bổ sung + lộ trình phát triển
- **Reverse matching**: Nhà tuyển dụng tìm ứng viên phù hợp cho JD

### 📊 Thống kê & Insights

- Thống kê thị trường (top kỹ năng, top ngành, phân bố lương...)
- AI phân tích xu hướng tuyển dụng
- Dashboard analytics

### 🔄 Quản lý phiên hội thoại

- Session-based: Mỗi cuộc trò chuyện có UUID riêng
- Lưu lịch sử hội thoại (tối đa 50 tin nhắn/session)
- Chatbot nhớ ngữ cảnh 5 lượt hội thoại gần nhất
- Tạo session mới / xoá lịch sử bất cứ lúc nào

---

## 🧠 Cách Chatbot hoạt động

### RAG Pipeline — Từ câu hỏi đến câu trả lời

```
┌─────────────────────────────────────────────────────────────────────┐
│                        NGƯỜI DÙNG HỎI                                │
│         "Tìm việc Python ở Đà Nẵng lương trên 20 triệu"            │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─ BƯỚC 1: RETRIEVAL ──────────────────────────────────────────────────┐
│                                                                       │
│  ┌─────────────────────┐       ┌────────────────────────────────┐    │
│  │   ChromaDB           │       │   PostgreSQL                    │    │
│  │   (Vector Store)     │       │   (Enrichment)                  │    │
│  │                      │       │                                  │    │
│  │  Query embedding     │       │  get_job_by_id() → full details │    │
│  │  → Semantic search   │──────▶│  get_job_stats() → thống kê     │    │
│  │  → Top-K similar     │       │  search_by_keyword() → bổ sung  │    │
│  │     jobs (k=3)       │       │                                  │    │
│  └─────────────────────┘       └────────────────────────────────┘    │
│                                                                       │
│  Output: Context string + Sources list                                │
│    📌 Senior Python Dev — FPT Software                                │
│       Công ty: FPT | Địa điểm: Đà Nẵng | Lương: 25-35 triệu        │
│       Kỹ năng: Python, FastAPI, PostgreSQL...                         │
│    📊 Thống kê: 6,338 jobs | Top skills: Python (890), Java (756)    │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
                            ▼
┌─ BƯỚC 2: AUGMENTED PROMPT ───────────────────────────────────────────┐
│                                                                       │
│  System Prompt (context_type="jobs"):                                 │
│    "Bạn là chuyên gia tuyển dụng và phân tích thị trường..."         │
│                                                                       │
│  + Conversation History (5 lượt gần nhất)                             │
│  + Retrieved Context (jobs + statistics)                              │
│  + User Question                                                      │
│                                                                       │
│  → Gộp thành 1 prompt hoàn chỉnh gửi đến LLM                        │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
                            ▼
┌─ BƯỚC 3: GENERATION ─────────────────────────────────────────────────┐
│                                                                       │
│  ┌──────────────────────────┐                                        │
│  │   Google Gemini 2.5      │                                        │
│  │   Flash                  │                                        │
│  │                          │                                        │
│  │  • Hiểu câu hỏi         │──▶  Câu trả lời tiếng Việt             │
│  │  • Tham chiếu context    │     + Gợi ý việc làm cụ thể           │
│  │  • Sinh câu trả lời     │     + Lời khuyên bổ sung               │
│  │  • Gợi ý thêm           │     + Confidence score                 │
│  └──────────────────────────┘                                        │
│                                                                       │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
                            ▼
┌─ BƯỚC 4: RESPONSE ───────────────────────────────────────────────────┐
│                                                                       │
│  {                                                                    │
│    "response": "Tôi tìm thấy 3 vị trí Python tại Đà Nẵng...",       │
│    "sources": [{"id":"42", "title":"...", "similarity": 0.92}],      │
│    "confidence_score": 0.89,                                          │
│    "session_id": "uuid-for-next-turn"                                │
│  }                                                                    │
│                                                                       │
│  → Lưu vào session history → Sẵn sàng cho câu hỏi tiếp theo         │
└──────────────────────────────────────────────────────────────────────┘
```

### Tại sao RAG?

| Phương pháp        | Hạn chế                                                | RAG giải quyết                                                   |
| :----------------- | :----------------------------------------------------- | :--------------------------------------------------------------- |
| **LLM thuần**      | Không biết dữ liệu việc làm thực tế, hay "hallucinate" | Cung cấp context thực từ database → trả lời chính xác            |
| **Keyword search** | Chỉ khớp từ khóa, bỏ sót kết quả liên quan             | Semantic search hiểu nghĩa → "AI" tìm được cả "Machine Learning" |
| **Full-text SQL**  | Không hiểu ngữ cảnh câu hỏi phức tạp                   | LLM hiểu ý định → biết lọc theo lương, kinh nghiệm, vị trí       |

---

## 🎭 4 Chế độ tư vấn

Chatbot có **4 chế độ chuyên biệt**, mỗi chế độ có system prompt riêng tối ưu cho từng mục đích:

### 🔍 `jobs` — Chuyên gia Tuyển dụng (mặc định)

```json
{
  "message": "Có vị trí nào remote cho React Developer không?",
  "context_type": "jobs"
}
```

> _System prompt:_ Bạn là **chuyên gia tuyển dụng** — nắm vững các vị trí đang tuyển, yêu cầu kỹ năng, xu hướng thị trường, mức lương và phúc lợi. Tham khảo dữ liệu thực để gợi ý công việc phù hợp.

**Chatbot sẽ:**

- Tìm kiếm trong database 6,338 việc làm
- Gợi ý 2-3 vị trí phù hợp nhất (kèm thông tin chi tiết)
- Phân tích xu hướng thị trường
- So sánh mức lương giữa các vị trí

---

### 📄 `cv` — Tư vấn viên CV

```json
{ "message": "CV fresher nên viết thế nào cho nổi bật?", "context_type": "cv" }
```

> _System prompt:_ Bạn là **chuyên gia tư vấn CV** — hiểu cấu trúc CV hiệu quả, kỹ năng trình bày, cách highlight kinh nghiệm. Đưa ra gợi ý cụ thể, thực thi được, khuyến khích tích cực.

**Chatbot sẽ:**

- Phân tích mạnh/yếu của CV
- Gợi ý cải thiện từng phần (skills, experience, education...)
- Ví dụ cụ thể cách viết từng mục
- Tips trình bày theo từng ngành

---

### 🔗 `matching` — Chuyên gia Matching

```json
{
  "message": "CV của tôi có kỹ năng Java, Spring Boot, 3 năm kinh nghiệm — phù hợp với công việc nào?",
  "context_type": "matching"
}
```

> _System prompt:_ Bạn là **chuyên gia ghép nối CV-Job** — phân tích mức độ phù hợp, xác định kỹ năng khớp/thiếu, đưa ra lộ trình phát triển cụ thể.

**Chatbot sẽ:**

- Đánh giá mức độ phù hợp (%) với từng vị trí
- Liệt kê kỹ năng khớp ✅ và thiếu ❌
- Gợi ý kỹ năng cần bổ sung
- Lộ trình phát triển cụ thể

---

### 🎯 `career` — Cố vấn Sự nghiệp

```json
{
  "message": "Tôi đang làm Tester, có nên chuyển sang Developer không?",
  "context_type": "career"
}
```

> _System prompt:_ Bạn là **cố vấn sự nghiệp** — giúp lập kế hoạch phát triển, xác định hướng đi, xử lý thách thức nghề nghiệp. Thân thiện, động viên, tôn trọng giá trị cá nhân.

**Chatbot sẽ:**

- Phân tích ưu/nhược của mỗi hướng đi
- Đưa ra lộ trình chuyển đổi thực tế
- Kế hoạch hành động cụ thể (timeline, skills cần học)
- Lời khuyên dựa trên xu hướng thị trường

---

## 🚀 Bắt đầu nhanh

### Yêu cầu

- **Python** 3.11+
- **PostgreSQL** 15+ _(tuỳ chọn — có thể chạy với SQLite)_
- **Google API Key** — [Lấy tại Google AI Studio](https://aistudio.google.com/apikey)

### Cài đặt & Chạy

```bash
# 1. Clone & setup
git clone https://github.com/betapcode17/PBL5_MATCHING_AI.git
cd AI
python -m venv venv && venv\Scripts\activate   # Windows
pip install -r requirements.txt

# 2. Cấu hình
echo GOOGLE_API_KEY=your-api-key > .env

# 3. (Tuỳ chọn) Import data vào PostgreSQL
echo DATABASE_URL=postgresql://user:pass@localhost:5432/it_job_db >> .env
python scripts/import_csv_to_postgres.py

# 4. Khởi chạy
uvicorn app.main:app --reload --port 8000
```

### Truy cập

| URL                         | Mô tả                                |
| :-------------------------- | :----------------------------------- |
| http://localhost:8000/chat  | 💬 **Giao diện Chat**                |
| http://localhost:8000/docs  | 📖 Swagger UI — Interactive API docs |
| http://localhost:8000/redoc | 📘 ReDoc                             |
| http://localhost:8000       | 🏠 API Status                        |

### Thử nhanh với `curl`

```bash
# Hỏi chatbot tìm việc
curl -X POST http://localhost:8000/api/chat/message \
  -H "Content-Type: application/json" \
  -d '{"message": "Tìm việc React ở Hà Nội", "context_type": "jobs"}'

# Tư vấn CV
curl -X POST http://localhost:8000/api/chat/message \
  -H "Content-Type: application/json" \
  -d '{"message": "Cách viết CV cho Backend Developer", "context_type": "cv"}'

# Tư vấn sự nghiệp
curl -X POST http://localhost:8000/api/chat/message \
  -H "Content-Type: application/json" \
  -d '{"message": "Lộ trình Junior đến Senior trong 3 năm?", "context_type": "career"}'
```

---

## 📡 API Endpoints

### 💬 Chat API — `/api/chat`

| Method | Endpoint                                    | Mô tả                                               |
| :----: | :------------------------------------------ | :-------------------------------------------------- |
| `POST` | `/api/chat/message`                         | **Gửi tin nhắn** → nhận phản hồi AI (RAG-augmented) |
| `GET`  | `/api/chat/history?session_id=...&limit=50` | Lịch sử hội thoại                                   |
| `POST` | `/api/chat/clear`                           | Xoá lịch sử session                                 |
| `GET`  | `/api/chat/new-session`                     | Tạo phiên mới                                       |
| `GET`  | `/api/chat/health`                          | Kiểm tra sức khoẻ (ChromaDB, Gemini, PostgreSQL)    |
| `GET`  | `/api/chat/info`                            | Thông tin chatbot & các chế độ hỗ trợ               |

<details>
<summary><b>📌 Chi tiết: POST /api/chat/message</b></summary>

**Request:**

```json
{
  "message": "string (1-2000 ký tự)",
  "session_id": "uuid (tuỳ chọn — nếu bỏ qua sẽ tạo session mới)",
  "context_type": "jobs | cv | matching | career (mặc định: jobs)"
}
```

**Response:**

```json
{
  "response": "Câu trả lời từ AI...",
  "session_id": "a1b2c3d4-e5f6-...",
  "timestamp": "2026-03-05T10:30:00",
  "sources": [
    {
      "id": "42",
      "title": "Python Developer",
      "company": "FPT Software",
      "location": "Đà Nẵng",
      "url": "https://...",
      "similarity": 0.92
    }
  ],
  "confidence_score": 0.85
}
```

**Luồng xử lý bên trong:**

1. Validate message (1–2000 ký tự)
2. Tạo hoặc tiếp tục session
3. RAG retrieval từ ChromaDB → enrich từ PostgreSQL
4. Build prompt = system prompt + history (5 turns) + context + user message
5. Gọi Google Gemini → generate response
6. Tính confidence score (trung bình similarity của sources)
7. Lưu vào history → Return response
</details>

<details>
<summary><b>📌 Chi tiết: GET /api/chat/health</b></summary>

```json
{
  "status": "ok",
  "timestamp": "2026-03-05T10:30:00",
  "version": "1.0.0",
  "services": {
    "chroma": true,
    "gemini": true,
    "database": true
  }
}
```

</details>

### 📄 CV Management — `/cv`

| Method | Endpoint                       | Mô tả                                                          |
| :----: | :----------------------------- | :------------------------------------------------------------- |
| `POST` | `/cv/upload`                   | Upload CV (PDF) → extract thông tin + index vào ChromaDB       |
| `GET`  | `/cv/list?page=1&page_size=10` | Danh sách CV (phân trang)                                      |
| `GET`  | `/cv/`                         | Tất cả CV với thông tin đã parse                               |
| `POST` | `/cv/delete`                   | Xoá CV                                                         |
| `GET`  | `/cv/{cv_id}/insights`         | Phân tích chất lượng CV (quality score, strengths, weaknesses) |
| `POST` | `/cv/improve?cv_id=1`          | Gợi ý cải thiện CV chi tiết                                    |

### 💼 Jobs — `/jobs`

| Method | Endpoint                    | Mô tả                            |
| :----: | :-------------------------- | :------------------------------- |
| `GET`  | `/jobs/?limit=100&offset=0` | Danh sách việc làm               |
| `POST` | `/jobs/search`              | Tìm kiếm với bộ lọc + AI ranking |
| `GET`  | `/jobs/analytics`           | Thống kê thị trường việc làm     |
| `POST` | `/jobs/analytics/insights`  | AI phân tích xu hướng            |

### 🔗 Matching — `/matching`

| Method | Endpoint                         | Mô tả                          |
| :----: | :------------------------------- | :----------------------------- |
| `POST` | `/matching/`                     | Match CV → Jobs (RAG pipeline) |
| `POST` | `/matching/apply`                | Ghi nhận ứng tuyển             |
| `GET`  | `/matching/applications/{cv_id}` | Lịch sử ứng tuyển              |

### 👥 Candidates — `/candidates`

| Method | Endpoint             | Mô tả                                 |
| :----: | :------------------- | :------------------------------------ |
| `POST` | `/candidates/search` | Reverse matching: Tìm ứng viên cho JD |

---

## 🏗 Kiến trúc RAG Pipeline

```
                         ┌────────────────────────────┐
                         │     Chat UI / Client App    │
                         │     (Web / Mobile / curl)   │
                         └─────────────┬──────────────┘
                                       │
                              POST /api/chat/message
                                       │
                         ┌─────────────▼──────────────┐
                         │       FastAPI Router         │
                         │      (routers/chat.py)       │
                         └─────────────┬──────────────┘
                                       │
                         ┌─────────────▼──────────────┐
                         │      ChatbotRAG Engine       │
                         │    (chat_service.py)         │
                         │                              │
                         │  ┌────────────────────────┐ │
                         │  │  Session Manager        │ │
                         │  │  Dict[uuid → history]   │ │
                         │  └────────────────────────┘ │
                         │                              │
                         │  ┌────────────────────────┐ │
                         │  │  _retrieve_context()    │ │
                         │  │  Semantic + Enrich      │─┼─────┐
                         │  └────────────────────────┘ │     │
                         │                              │     │
                         │  ┌────────────────────────┐ │     │
                         │  │  _build_system_prompt() │ │     │
                         │  │  (4 context types)      │ │     │
                         │  └────────────────────────┘ │     │
                         │                              │     │
                         │  ┌────────────────────────┐ │     │
                         │  │  LLMService.generate()  │ │     │
                         │  │  Google Gemini 2.5      │ │     │
                         │  └────────────────────────┘ │     │
                         └──────────────────────────────┘     │
                                                              │
                    ┌──────────────────┐   ┌──────────────────▼─┐
                    │   ChromaDB        │   │   PostgreSQL        │
                    │   Vector Store    │   │   it_job_db         │
                    │                   │   │                     │
                    │ Collection: jobs  │   │  6,338 JobPost      │
                    │   6,338 documents │   │  2,906 Company      │
                    │   768-dim vectors │   │  6,371 Skill        │
                    │                   │   │  344 Category       │
                    │ Collection: cvs   │   │  36,982 JobPostSkill│
                    │   (dynamic)       │   │                     │
                    │                   │   │  + Statistics API   │
                    │ Embedding model:  │   │  + Full-text search │
                    │ text-embedding-004│   │  + Data enrichment  │
                    └──────────────────┘   └─────────────────────┘
```

---

## ⚙️ Services chi tiết

### `ChatbotRAG` — Trái tim của Chatbot

```python
class ChatbotRAG:
    """RAG-powered chatbot engine"""

    # Khởi tạo
    def __init__(self, collection_name="jobs", k_documents=3, enable_rag=True)

    # Session management
    def create_session() -> str                    # Tạo UUID session mới
    def get_history(session_id, limit=50)          # Lấy lịch sử chat
    def clear_history(session_id) -> bool          # Xoá hội thoại

    # Core logic
    def chat(message, session_id, context_type, use_rag) -> ChatResponse
    #   │
    #   ├── _retrieve_context(query)              # ChromaDB search + PG enrich
    #   ├── _build_system_prompt(context_type)     # Chọn prompt theo chế độ
    #   ├── _build_conversation_context(5 turns)   # Gộp lịch sử gần nhất
    #   ├── LLMService.generate_response(prompt)   # Gọi Gemini
    #   └── _calculate_confidence(sources)         # Tính confidence score
```

### `RetrievalService` — Tìm kiếm Semantic

```python
class RetrievalService:
    def retrieve(query, k=3, filters=None) -> List[RetrievedDocument]
    #   → ChromaDB similarity_search_with_relevance_scores
    #   → Trả về documents + khoảng cách (similarity)

    def retrieve_by_metadata(field, value, k=10)    # Lọc theo metadata
    def get_context_string(query, k=3) -> str       # Format context cho LLM
    def retrieve_similar_jobs(job_id, k=5)          # Tìm jobs tương tự
```

### `LLMService` — Google Gemini Wrapper

```python
class LLMService:
    def generate_response(message, system_prompt=None) -> str
    #   → Gọi Gemini với system prompt + user message

    def generate_with_context(message, context, system_prompt) -> str
    #   → RAG: inject context vào prompt

    def extract_entities(text) -> dict
    #   → Trích xuất: job_titles, locations, skills, salary_range...

    def rate_match(cv_summary, job_description) -> dict
    #   → {match_percentage, strengths, gaps, recommendation}
```

### `APIKeyManager` — Xoay vòng API Keys

```python
class APIKeyManager:
    """Round-robin rotation giữa 1-9 Google API keys"""
    # Tự động detect keys: GOOGLE_API_KEY, GOOGLE_API_KEY_1, ..._9
    # get_next_api_key() → key tiếp theo trong round-robin
    # Tránh vượt quota của Google AI
```

---

## 📦 Data Models

### Request & Response

```python
# === Chat ===
class ChatMessage:
    message: str           # 1-2000 ký tự
    session_id: str?       # Tiếp tục hội thoại (tuỳ chọn)
    context_type: str?     # "jobs" | "cv" | "matching" | "career"

class ChatResponse:
    response: str          # Câu trả lời AI
    session_id: str        # UUID phiên hội thoại
    timestamp: datetime    # Thời gian phản hồi
    sources: List[Source]  # Nguồn tham khảo (jobs matched)
    confidence_score: float # 0.0 → 1.0 (độ tin cậy)

class ChatHistoryItem:
    role: "user" | "assistant"
    content: str
    timestamp: datetime
    sources: List[Source]?

# === CV ===
class CVInfo:
    name, email, phone: str
    career_objective: str
    skills: List[str]
    education: List[Education]
    experience: List[Experience]

class CVInsights:
    quality_score: float    # 0 → 10
    completeness: {...}     # has_portfolio, has_certifications, missing_sections
    market_fit: {...}       # skill_match_rate, experience_level, salary_range
    strengths: List[str]
    weaknesses: List[str]

# === Matching ===
class MatchedJob:
    job_id, job_title, company_name: str
    match_score: float      # 0.0 → 1.0
    matched_skills: List[str]
    explanation: MatchExplanation   # skills, experience, education analysis
    why_match: str                  # Giải thích tiếng Việt
```

---

## 🗄 Database & Vector Store

### PostgreSQL — `it_job_db` (Primary)

| Table            |  Rows  | Mô tả                                                      |
| :--------------- | :----: | :--------------------------------------------------------- |
| **JobPost**      | 6,338  | Việc làm (title, description, salary, location, skills...) |
| **Company**      | 2,906  | Công ty (name, size, industry, website)                    |
| **Skill**        | 6,371  | Kỹ năng IT (name, type)                                    |
| **JobPostSkill** | 36,982 | Mapping job ↔ skill                                        |
| **Category**     |  344   | Danh mục việc làm                                          |
| **JobType**      |   6    | Loại hình (Toàn thời gian, Bán thời gian...)               |
| **Employee**     |   2    | Nhà tuyển dụng                                             |

### ChromaDB — Vector Store

| Collection | Documents | Embedding                    | Vai trò                      |
| :--------- | :-------- | :--------------------------- | :--------------------------- |
| `jobs`     | 6,338     | text-embedding-004 (768-dim) | Semantic search cho chatbot  |
| `cvs`      | Dynamic   | text-embedding-004 (768-dim) | Reverse matching (Job → CVs) |

**Document format trong ChromaDB:**

```
JOB_ID: 42
Senior Python Developer
Mô tả: Phát triển hệ thống backend sử dụng Python...
Yêu cầu: 3+ năm kinh nghiệm Python, FastAPI, PostgreSQL...
Kỹ năng: ["Python", "FastAPI", "PostgreSQL", "Docker"]
```

### SQLite — Fallback & Cache

```
cv_store         → Lưu CV đã upload (PDF + parsed JSON)
job_store        → Cache jobs (fallback khi không có PostgreSQL)
applications     → Tracking ứng tuyển
match_logs       → Cache kết quả matching (TTL: 1 giờ)
cv_insights      → Cache phân tích CV
```

---

## 📁 Cấu trúc thư mục

```
AI/
├── 📄 pyproject.toml                 # Metadata & dependencies (Poetry)
├── 📄 requirements.txt               # Pip dependencies
│
├── 📂 app/                           # ========== FASTAPI APPLICATION ==========
│   ├── 📄 main.py                    # Entry point — startup preload, CORS, routers
│   ├── 📄 config.py                  # DATABASE_URL, API keys, paths
│   ├── 📄 dependencies.py            # FastAPI dependency injection
│   │
│   ├── 📂 routers/                   # --- API Endpoints ---
│   │   ├── 📄 chat.py                # 💬 /api/chat/* — Chatbot endpoints
│   │   ├── 📄 cv.py                  # 📄 /cv/* — Upload, insights, improve
│   │   ├── 📄 jobs.py                # 💼 /jobs/* — Search, analytics
│   │   ├── 📄 matching.py            # 🔗 /matching/* — CV↔Job matching
│   │   ├── 📄 candidates.py          # 👥 /candidates/* — Reverse matching
│   │   └── 📄 utils.py               # 🔧 Utility endpoints
│   │
│   ├── 📂 services/                  # --- Business Logic ---
│   │   ├── 📄 chat_service.py        # 🧠 ChatbotRAG — core chatbot engine
│   │   ├── 📄 llm_service.py         # 🤖 LLMService — Gemini wrapper
│   │   ├── 📄 retrieval_service.py   # 🔍 RetrievalService — ChromaDB search
│   │   ├── 📄 rag_matching.py        # 🔗 RAG matching pipeline
│   │   ├── 📄 ai_analysis.py         # 📊 CV insights & improvements
│   │   ├── 📄 chroma_utils.py        # 💾 ChromaDB operations (preload, index)
│   │   ├── 📄 pg_database.py         # 🐘 PostgreSQL connector & queries
│   │   ├── 📄 db_utils.py            # 📦 SQLite operations (6 tables)
│   │   ├── 📄 api_key_manager.py     # 🔑 API key rotation (1-9 keys)
│   │   ├── 📄 candidate_matching.py  # 👥 Reverse matching (Job → CVs)
│   │   ├── 📄 match_explain.py       # 💡 Match explanation normalizer
│   │   └── 📄 rag_helpers.py         # 🛠️ RAG utility functions
│   │
│   ├── 📂 models/                    # --- Pydantic Models ---
│   │   ├── 📄 chat.py                # ChatMessage, ChatResponse, Session...
│   │   ├── 📄 core.py                # CVInfo, JobInfo, MatchedJob, Filters
│   │   └── 📄 responses.py           # API response schemas
│   │
│   ├── 📂 prompts/                   # --- System Prompts ---
│   │   ├── 📄 chat_system_prompt.py  # 4 chế độ: jobs, cv, matching, career
│   │   ├── 📄 qa_prompt.py           # RAG matching prompt
│   │   ├── 📄 rewrite_prompt.py      # Query rewrite cho semantic search
│   │   ├── 📄 cv_analysis_prompt.py  # CV quality analysis
│   │   ├── 📄 cv_improvement_prompt.py # CV improvement suggestions
│   │   └── 📄 chart_insights_prompt.py # Analytics AI insights
│   │
│   ├── 📂 utils/                     # --- Helpers ---
│   │   ├── 📄 pdf_parser.py          # PDF → text → Gemini extract JSON
│   │   ├── 📄 date_utils.py          # Vietnamese date normalization
│   │   └── 📄 validators.py          # Input validation
│   │
│   ├── 📂 data/                      # --- Data Files ---
│   │   ├── 📄 jobs_vietnamese.csv    # 3,237 jobs (Vietnamese market)
│   │   ├── 📄 job_descriptions.csv   # 3,101 jobs (ITViec)
│   │   └── 📄 jobs_processed.jsonl   # Processed jobs
│   │
│   ├── 📂 db/chroma_db/             # ChromaDB persistence
│   └── 📂 middleware/error_handler.py # Global exception handler
│
├── 📂 static/                        # Chat UI (HTML/CSS/JS)
├── 📂 scripts/                       # Import & utility scripts
├── 📂 cv_screener/                   # Batch CV screening module
├── 📂 docs/                          # Documentation
└── 📂 tests/                         # Test files
```

---

## ⚙️ Cấu hình

Tạo file `.env` tại thư mục gốc:

```env
# ============ BẮT BUỘC ============
GOOGLE_API_KEY=your-primary-api-key

# ============ DATABASE ============
# PostgreSQL (nếu không cấu hình → dùng SQLite fallback)
DATABASE_URL=postgresql://postgres:password@localhost:5432/it_job_db

# ============ API KEY ROTATION ============
# Hỗ trợ 1-9 keys, round-robin tự động xoay vòng tránh quota
GOOGLE_API_KEY_1=key-1
GOOGLE_API_KEY_2=key-2
GOOGLE_API_KEY_3=key-3

# ============ CHATBOT ============
DEFAULT_CONTEXT=jobs          # Chế độ mặc định: jobs | cv | matching | career
RAG_K_DOCUMENTS=3             # Số documents retrieve cho mỗi câu hỏi
ENABLE_RAG=True               # Bật/tắt RAG pipeline
MAX_HISTORY_LENGTH=50         # Số tin nhắn tối đa mỗi session

# ============ LLM ============
LLM_MODEL=gemini-pro
LLM_TEMPERATURE=0.7           # 0.0 (chính xác) → 1.0 (sáng tạo)
LLM_MAX_TOKENS=1024
EMBEDDING_MODEL=text-embedding-004

# ============ SERVER ============
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000
```

---

## 📊 Thông số hệ thống

| Thông số          | Giá trị        | Ghi chú                 |
| :---------------- | :------------- | :---------------------- |
| **Jobs trong DB** | **6,338**      | 3,237 VN + 3,101 ITViec |
| Companies         | 2,906          | Deduplicated            |
| Skills            | 6,371          | Technical + soft skills |
| Embedding dim     | 768            | text-embedding-004      |
| ChromaDB top-k    | 3              | Có thể tăng lên 10      |
| Chat history      | 50 msg/session | Context window: 5 turns |
| Message max       | 2,000 ký tự    | Validation              |
| API keys          | 1–9            | Round-robin rotation    |
| Confidence score  | 0.0–1.0        | Trung bình similarity   |
| Response          | Tiếng Việt     | Hỗ trợ đa ngôn ngữ      |

---

## 🔬 Mở rộng: CV Screener Module

Module batch matching độc lập nằm trong `cv_screener/`, sử dụng **Sentence-BERT** cho tốc độ cao:

- **4 tiêu chí**: Kinh nghiệm (25%) + Kỹ năng (25%) + Dự án (25%) + Tính cách (25%)
- **Batch processing**: 250,000 matches trong ~17 giây
- **Web scraping**: Crawler Selenium + Scrapy
- **Fine-tuning**: Fine-tune model trên dữ liệu thực tế

```bash
cd cv_screener
python run_simple_pipeline.py    # Pipeline cơ bản
streamlit run app.py             # Giao diện web
```

---

## 🧪 Testing

```bash
# Unit tests
pip install pytest pytest-asyncio httpx
pytest tests/ -v

# Thử nhanh chatbot
curl -X POST http://localhost:8000/api/chat/message \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, bạn có thể giúp gì cho tôi?"}'

# Health check
curl http://localhost:8000/api/chat/health
```

---

## 🤝 Đóng góp

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

---

## 👥 Tác giả

| Tên             | Vai trò   | GitHub                                         |
| :-------------- | :-------- | :--------------------------------------------- |
| **betapcode17** | Developer | [@betapcode17](https://github.com/betapcode17) |

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for more information.

---

<div align="center">

**⭐ Nếu dự án hữu ích, hãy cho một Star!**

Made with ❤️ for PBL5 — Đại học Bách Khoa Đà Nẵng

</div>
