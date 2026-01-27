# 🧠 APP - Ứng dụng FastAPI chính

**Vị trí:** `e:\HOCKI6\PBL5\AI\app\`

## Mục đích

Ứng dụng backend FastAPI để xử lý:

- 📄 Upload & phân tích CV
- 📋 Quản lý Job postings
- 🎯 Matching CV-Job
- 💬 Q&A với LLM
- 🔄 RAG pipeline

## Cấu trúc

```
app/
├── main.py              # 🚀 FastAPI app, routes chính
├── config.py            # ⚙️ Cấu hình (API keys, paths)
├── dependencies.py      # 🔗 Dependency injection
├── models/              # 📦 Pydantic schemas
├── routers/             # 🛣️ API endpoints
├── services/            # 🧠 Business logic
├── prompts/             # 📝 LLM prompts
├── utils/               # 🔧 Tiện ích
└── data/                # 📊 Dữ liệu
```

## Các Endpoint chính

| Method | Path                 | Mục đích               |
| ------ | -------------------- | ---------------------- |
| POST   | `/cv/upload`         | Upload CV (PDF/DOC)    |
| GET    | `/cv/{id}`           | Lấy thông tin CV       |
| POST   | `/cv/analyze`        | Phân tích CV với LLM   |
| POST   | `/cv/improve`        | Gợi ý cải thiện CV     |
| GET    | `/jobs`              | Danh sách jobs         |
| POST   | `/jobs`              | Tạo job mới            |
| POST   | `/matching/search`   | Tìm CV phù hợp cho Job |
| POST   | `/candidates/filter` | Lọc candidates         |

## Cách chạy

```bash
cd e:\HOCKI6\PBL5\AI
python -m app.main
# Hoặc: uvicorn app.main:app --reload

# Truy cập API docs: http://localhost:8000/docs
```

## Dependency

Xem `requirements.txt` ở thư mục gốc:

- fastapi, uvicorn
- pydantic
- openai (LLM API)
- chromadb (vector DB)
- pandas, numpy
- torch, sentence-transformers

## Mở rộng

**Thêm endpoint mới:**

1. Tạo file `routers/new_feature.py`
2. Thêm routes trong đó
3. Import vào `main.py`: `from app.routers import new_feature`
4. Include router: `app.include_router(new_feature.router)`

**Thêm service mới:**

1. Tạo file `services/new_service.py`
2. Viết business logic
3. Import và sử dụng trong routers

## Lưu ý

- API keys được lưu trong `config.py`
- Database connections quản lý qua `dependencies.py`
- Models (Pydantic) định nghĩa request/response schema
- Services không phụ thuộc vào HTTP framework

---

**Tìm hiểu thêm:** Xem `1_Huong_Dan_Su_Dung/README.md`
