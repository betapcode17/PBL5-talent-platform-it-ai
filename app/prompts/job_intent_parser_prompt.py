"""
Job Intent Parser Prompt
Prompt for parsing user queries to detect job search intent and extract parameters
"""

def get_job_parser_prompt() -> str:
    """
    Get structured prompt for Ollama to parse job queries.
    Instructs LLM to analyze user queries and return structured JSON with job search parameters.
    
    Returns:
        str: System prompt for job intent parsing
    """
    return """ROLE: Bạn là một trợ lý phân tích truy vấn tìm việc chuyên nghiệp và chính xác cao.

TASK: Phân tích câu hỏi/truy vấn của user và xác định:
1. Liệu đó có phải là truy vấn tìm việc không
2. Loại công cụ cần sử dụng (search, details, company)
3. Chi tiết tìm kiếm: danh mục, vị trí, lương, từ khóa, công ty

OUTPUT FORMAT: Trả về STRICTLY JSON ONLY, không có text, giải thích, hay markdown. Chỉ JSON hợp lệ.

JSON STRUCTURE:
{
  "has_job_query": boolean,           # true = truy vấn về việc, false = không liên quan
  "tool": string | null,              # "search_jobs" | "get_job_details" | "get_company_jobs" | null
  "category": string | null,          # "Backend" | "Frontend" | "FullStack" | "DevOps" | "QA" | "Mobile" | "Data" | "AI/ML" | "DevOps" | null
  "location": string | null,          # "Hanoi" | "HCM" | "Da Nang" | "Hai Phong" | null
  "keywords": string | null,          # mô tả công việc/kỹ năng tìm kiếm
  "salary_min": string | null,        # "20m" | "30k" | "50m" | null (m = triệu, k = ngàn)
  "job_id": number | null,            # ID công việc nếu được mention
  "company_name": string | null,      # Tên công ty nếu được mention
  "limit": number,                    # Số kết quả trả về (1-50, mặc định 20)
  "page": number,                     # Trang kết quả (mặc định 1)
  "confidence": number                # Độ tin cậy (0.0-1.0): 1.0 = rất chắc chắn, 0.0 = không chắc
}

PARSING RULES:
1. JOB DETECTION: Tìm các từ khóa như "tìm", "tuyển", "job", "việc", "position", "hiring", "backend", "frontend"
   - Nếu không có từ khóa việc → has_job_query = false, tool = null
   
2. TOOL SELECTION:
   - Nếu user hỏi chi tiết công việc cụ thể (ID, "job nào", "công việc này") → "get_job_details"
   - Nếu user hỏi việc của công ty cụ thể → "get_company_jobs"
   - Nếu user search công việc chung (category, location, kỹ năng) → "search_jobs"
   
3. CATEGORY MAPPING: Chuyển đổi chính xác:
   - "python", "django", "backend" → "Backend"
   - "react", "vue", "angular" → "Frontend"
   - "devops", "ci/cd", "infrastructure" → "DevOps"
   - "test", "qa", "kiểm thử" → "QA"
   - "ai", "machine learning", "data science" → "AI/ML"
   
4. LOCATION NORMALIZATION:
   - "hà nội", "ha noi" → "Hanoi"
   - "sài gòn", "hcm", "tp.hcm" → "HCM"
   - "đà nẵng" → "Da Nang"
   
5. SALARY PARSING:
   - "20 triệu" → "20m"
   - "30,000" → "30k"
   - Nếu có khoảng lương, lấy mức tối thiểu (minimum)
   
6. KEYWORDS: Trích xuất các từ khóa chính về kỹ năng, công nghệ (nếu không có category cụ thể)

7. CONFIDENCE SCORING:
   - 1.0 = "Chi tiết công việc ID 123" (rõ ràng)
   - 0.8-0.9 = "Backend job ở Hà Nội" (rõ ràng)
   - 0.6-0.7 = "Công việc lập trình" (chung chung)
   - 0.3-0.5 = "Có việc làm không?" (mơ hồ)
   - 0.0 = "Thời tiết thế nào?" (không phải tìm việc)

EXAMPLES:
1. "Tìm Backend engineer ở Hà Nội lương 25m"
   → {"has_job_query": true, "tool": "search_jobs", "category": "Backend", "location": "Hanoi", "salary_min": "25m", "limit": 20, "page": 1, "confidence": 0.95}

2. "Cho tôi xem chi tiết công việc số 6283"
   → {"has_job_query": true, "tool": "get_job_details", "job_id": 6283, "confidence": 1.0}

3. "Công ty FPT có tuyển dụng gì không?"
   → {"has_job_query": true, "tool": "get_company_jobs", "company_name": "FPT", "confidence": 0.9}

4. "Frontend React jobs ở HCM lương 20-30 triệu, limit 50"
   → {"has_job_query": true, "tool": "search_jobs", "category": "Frontend", "keywords": "React", "location": "HCM", "salary_min": "20m", "limit": 50, "page": 1, "confidence": 0.95}

5. "Hôm nay trời như thế nào?"
   → {"has_job_query": false, "tool": null, "confidence": 1.0}

6. "Python developer jobs, show 30 results"
   → {"has_job_query": true, "tool": "search_jobs", "category": "Backend", "keywords": "Python", "limit": 30, "page": 1, "confidence": 0.9}

CRITICAL REQUIREMENTS:
✓ Output ONLY valid JSON, nothing else
✓ Không có markdown code blocks (```)
✓ Không có giải thích hay text thêm
✓ Tất cả string values phải trong dấu ngoặc kép
✓ Boolean values: true/false (không "True"/"False")
✓ Null không bọc trong dấu ngoặc kép
✓ Confidence score luôn từ 0.0 đến 1.0"""


def get_job_keywords() -> list:
    """
    Get list of keywords to detect job-related queries.
    Used for quick pre-filtering before LLM parsing.
    More comprehensive to catch various job search intent variations.
    
    Returns:
        list: Keywords indicating job search intent
    """
    return [
        # Vietnamese job-related words
        "tìm", "việc", "tuyển", "tuyển dụng", "nhân viên", "ứng viên", "hồ sơ",
        "công ty", "công việc", "vị trí", "chức vụ", "bộ phận", "phòng ban",
        "lương", "tiền lương", "thù lao", "hợp đồng", "bán thời gian", "toàn thời gian",
        "kinh nghiệm", "kỹ năng", "yêu cầu", "đòi hỏi", "năng lực",
        "hà nội", "hcm", "sài gòn", "đà nẵng", "hải phòng",
        "remote", "offsite", "hybrid", "địa điểm", "làm việc",
        "startup", "công nghệ", "fintech", "e-commerce",
        
        # English job-related words
        "job", "position", "hiring", "recruit", "candidate", "applicant",
        "company", "workplace", "salary", "wage", "compensation", "benefit",
        "contract", "fulltime", "parttime", "remote", "onsite",
        "experience", "skill", "requirement", "qualification",
        "backend", "frontend", "fullstack", "devops", "qa", "mobile", "data",
        "python", "java", "javascript", "react", "node", "django", "spring",
        "machine learning", "ai", "data science", "cloud", "aws", "azure",
        
        # Vietnamese tech terms
        "lập trình", "kỹ sư", "developer", "engineer", "programmer",
        "phát triển", "phần mềm", "ứng dụng", "website", "mobile app",
        "database", "cơ sở dữ liệu", "hệ thống", "infrastructure",
        "test", "kiểm thử", "quality", "chất lượng", "bug", "lỗi"
    ]


def get_job_categories() -> dict:
    """
    Get mapping of keywords to job categories.
    Used in regex-based fallback parsing.
    More comprehensive to catch various technology and job type keywords.
    
    Returns:
        dict: Mapping of keywords to job categories
    """
    return {
        # Backend technologies
        "backend": "Backend", "python": "Backend", "java": "Backend",
        "php": "Backend", "c#": "Backend", "csharp": "Backend", "dotnet": "Backend",
        "golang": "Backend", "rust": "Backend", "ruby": "Backend",
        "nodejs": "Backend", "node.js": "Backend", "express": "Backend",
        "django": "Backend", "flask": "Backend", "spring": "Backend",
        "laravel": "Backend", "asp.net": "Backend",
        
        # Frontend technologies
        "frontend": "Frontend", "react": "Frontend", "vue": "Frontend",
        "angular": "Frontend", "javascript": "Frontend", "typescript": "Frontend",
        "html": "Frontend", "css": "Frontend", "nextjs": "Frontend",
        "nuxt": "Frontend", "svelte": "Frontend", "web": "Frontend",
        
        # Full stack
        "fullstack": "FullStack", "full-stack": "FullStack", "full stack": "FullStack",
        "mern": "FullStack", "mean": "FullStack",
        
        # DevOps
        "devops": "DevOps", "devop": "DevOps", "sre": "DevOps",
        "docker": "DevOps", "kubernetes": "DevOps", "k8s": "DevOps",
        "ci/cd": "DevOps", "cicd": "DevOps", "infrastructure": "DevOps",
        "cloud": "DevOps", "aws": "DevOps", "azure": "DevOps", "gcp": "DevOps",
        
        # QA
        "qa": "QA", "qc": "QA", "test": "QA", "testing": "QA",
        "kiểm thử": "QA", "automation": "QA", "autotest": "QA",
        "selenium": "QA", "appium": "QA", "cypress": "QA",
        
        # Mobile
        "mobile": "Mobile", "ios": "Mobile", "android": "Mobile",
        "flutter": "Mobile", "react native": "Mobile", "swift": "Mobile",
        "kotlin": "Mobile", "app": "Mobile",
        
        # Data
        "data": "Data", "data science": "Data", "data scientist": "Data",
        "machine learning": "Data", "ml": "Data", "ai": "Data",
        "artificial intelligence": "Data", "analytics": "Data",
        "sql": "Data", "spark": "Data", "hadoop": "Data",
    }


def get_job_locations() -> dict:
    """
    Get mapping of Vietnamese location names to English equivalents.
    Used in regex-based fallback parsing.
    More comprehensive to catch various spelling variations and abbreviations.
    
    Returns:
        dict: Mapping of location keywords to normalized names
    """
    return {
        # Hanoi variations
        "hà nội": "Hanoi", "hanoi": "Hanoi", "ha noi": "Hanoi",
        "ha.noi": "Hanoi", "h.noi": "Hanoi",
        
        # HCM variations
        "sài gòn": "HCM", "saigon": "HCM", "sai gon": "HCM",
        "hcm": "HCM", "tp.hcm": "HCM", "tp hcm": "HCM",
        "tphcm": "HCM", "hồ chí minh": "HCM", "ho chi minh": "HCM",
        
        # Da Nang variations
        "đà nẵng": "Da Nang", "da nang": "Da Nang", "danang": "Da Nang",
        "da.nang": "Da Nang",
        
        # Hai Phong variations
        "hải phòng": "Hai Phong", "hai phong": "Hai Phong",
        "haiphong": "Hai Phong", "hai.phong": "Hai Phong",
        
        # Other cities
        "huế": "Hue", "hue": "Hue",
        "cần thơ": "Can Tho", "can tho": "Can Tho", "cantho": "Can Tho",
        "quảng ninh": "Quang Ninh", "quang ninh": "Quang Ninh",
        "bắc ninh": "Bac Ninh", "bac ninh": "Bac Ninh",
        
        # Special locations
        "remote": "Remote", "online": "Remote", "làm việc từ nhà": "Remote",
        "offline": "Onsite", "onsite": "Onsite",
    }
