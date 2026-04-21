"""
Query Normalizer - Chuẩn hóa input user trước khi parse
Purpose: Convert messy user input into clean, standardized format
để improve parsing success rate
"""

import re
from typing import Dict, Optional, List


class QueryNormalizer:
    """Normalize user queries để dễ dàng parse"""
    
    # Mapping từ viết tắt/sai chính tả sang từ chuẩn
    ABBREVIATIONS = {
        # Vị trí
        "hn": "hà nội", "hcm": "tp.hcm", "tphcm": "tp.hcm", 
        "sai gon": "sài gòn", "saigon": "sài gòn",
        "da nang": "đà nẵng", "danang": "đà nẵng",
        "hai phong": "hải phòng", "haiphong": "hải phòng",
        
        # Công nghệ & Ngôn ngữ
        "py": "python", "js": "javascript", "ts": "typescript",
        "fe": "frontend", "be": "backend", "fs": "fullstack",
        "cicd": "ci/cd", "k8s": "kubernetes",
        "ml": "machine learning", "ai": "artificial intelligence",
        "db": "database", "qa": "quality assurance",
        
        # Hình thức làm việc
        "ft": "fulltime", "pt": "parttime", "wfh": "remote",
        "onsite": "office", "remote": "remote",
        
        # Mức độ
        "jr": "junior", "sr": "senior", "mid": "middle", "intern": "internship",
        "entry": "entry level", "expert": "senior",
        
        # Lương
        "m": "triệu", "k": "ngàn", "tr": "triệu",
    }
    
    # Keywords cho từng category
    CATEGORY_KEYWORDS = {
        "Backend": ["backend", "python", "java", "nodejs", "php", "golang", "django", "spring", "api"],
        "Frontend": ["frontend", "react", "vue", "angular", "javascript", "typescript", "css", "html"],
        "FullStack": ["fullstack", "full-stack", "mern", "mean", "mean stack"],
        "DevOps": ["devops", "docker", "kubernetes", "ci/cd", "infrastructure", "cloud", "aws", "azure"],
        "QA": ["qa", "test", "automation", "selenium", "testing"],
        "Mobile": ["mobile", "ios", "android", "flutter", "react native", "swift"],
        "Data": ["data", "machine learning", "ai", "data science", "analytics", "spark"],
    }
    
    # Location variations
    LOCATION_MAP = {
        "hà nội": "Hanoi",
        "ha noi": "Hanoi",
        "hanoi": "Hanoi",
        "tp.hcm": "HCM",
        "tphcm": "HCM",
        "sài gòn": "HCM",
        "saigon": "HCM",
        "hcm": "HCM",
        "đà nẵng": "Da Nang",
        "da nang": "Da Nang",
        "danang": "Da Nang",
        "hải phòng": "Hai Phong",
        "hai phong": "Hai Phong",
        "haiphong": "Hai Phong",
    }
    
    @staticmethod
    def normalize(query: str) -> Dict[str, any]:
        """
        Normalize query and extract structured info
        
        Returns:
            Dict with:
            - original: original query
            - normalized: cleaned query
            - extracted: extracted parts (location, keywords, etc.)
            - standardized: standardized form for parsing
        """
        original = query.strip()
        normalized = QueryNormalizer._clean_text(original)
        normalized = QueryNormalizer._expand_abbreviations(normalized)
        
        extracted = QueryNormalizer._extract_parts(normalized)
        standardized = QueryNormalizer._build_standardized_form(extracted, normalized)
        
        return {
            "original": original,
            "normalized": normalized,
            "extracted": extracted,
            "standardized": standardized,
        }
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean text: remove extra spaces, special chars, etc."""
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Normalize Vietnamese diacritics
        text = text.lower()
        
        # Remove trailing punctuation
        text = re.sub(r'[?!;,]+$', '', text)
        
        return text
    
    @staticmethod
    def _expand_abbreviations(text: str) -> str:
        """Replace abbreviations with full words"""
        text_lower = text.lower()
        
        for abbr, full in QueryNormalizer.ABBREVIATIONS.items():
            # Match whole word only
            pattern = r'\b' + re.escape(abbr) + r'\b'
            text_lower = re.sub(pattern, full, text_lower, flags=re.IGNORECASE)
        
        return text_lower
    
    @staticmethod
    def _extract_parts(text: str) -> Dict[str, Optional[str]]:
        """Extract structured parts from normalized text"""
        result = {
            "location": None,
            "category": None,
            "keywords": [],
            "salary_min": None,
            "job_id": None,
            "level": None,
            "employment_type": None,
        }
        
        text_lower = text.lower()
        
        # Extract location
        for vn_loc, en_loc in QueryNormalizer.LOCATION_MAP.items():
            if vn_loc in text_lower:
                result["location"] = en_loc
                break
        
        # Extract category
        for category, keywords in QueryNormalizer.CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    result["category"] = category
                    break
            if result["category"]:
                break
        
        # Extract job ID (if pattern like "id 123", "job 123", "#123")
        job_id_patterns = [
            r'(?:id|job|#)\s*(\d{3,5})',  # id 123, job 123, #123
            r'(?:công việc|chi tiết)\s+(?:số|id)?\s*(\d{3,5})',  # công việc 123
        ]
        for pattern in job_id_patterns:
            match = re.search(pattern, text_lower)
            if match:
                result["job_id"] = int(match.group(1))
                break
        
        # Extract salary (20m, 30k, 20-30 triệu, etc.)
        salary_patterns = [
            r'(\d+)\s*-\s*(\d+)\s*(?:triệu|m)',  # 20-30 triệu
            r'(\d+)\s*(?:triệu|m)',  # 20 triệu
            r'(\d+)\s*(?:ngàn|k)',  # 30k
        ]
        for pattern in salary_patterns:
            match = re.search(pattern, text_lower)
            if match:
                result["salary_min"] = f"{match.group(1)}m" if "triệu" in text_lower or "m" in text_lower else f"{match.group(1)}k"
                break
        
        # Extract level (junior, senior, intern, etc.)
        level_keywords = {
            "junior": ["junior", "jr", "fresher", "new graduate"],
            "middle": ["middle", "mid", "experienced"],
            "senior": ["senior", "sr", "expert", "lead"],
            "intern": ["intern", "internship", "thực tập"],
        }
        for level, keywords in level_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    result["level"] = level
                    break
            if result["level"]:
                break
        
        # Extract employment type
        employment_types = {
            "fulltime": ["fulltime", "full-time", "toàn thời gian"],
            "parttime": ["parttime", "part-time", "bán thời gian"],
            "remote": ["remote", "wfh", "work from home"],
            "hybrid": ["hybrid", "mixed"],
            "office": ["office", "onsite", "tại văn phòng"],
        }
        for emp_type, keywords in employment_types.items():
            for keyword in keywords:
                if keyword in text_lower:
                    result["employment_type"] = emp_type
                    break
            if result["employment_type"]:
                break
        
        # Extract keywords (remaining important words)
        # Remove stopwords
        stopwords = {"tìm", "việc", "công ty", "job", "position", "là", "được", "có", "với", 
                     "và", "hoặc", "không", "gì", "trong", "ở", "tại", "at", "in", "to"}
        words = text_lower.split()
        result["keywords"] = [w for w in words 
                            if len(w) > 2 and w not in stopwords 
                            and not re.match(r'\d+', w)]
        
        return result
    
    @staticmethod
    def _build_standardized_form(extracted: Dict, normalized: str) -> str:
        """
        Build a standardized query form for better parsing
        
        Example:
        Input: "tìm be ở hn lương 25m jr fulltime"
        Standardized: "search_jobs category:Backend location:Hanoi salary_min:25m level:junior employment_type:fulltime"
        """
        parts = []
        
        # Determine intent
        if extracted.get("job_id"):
            parts.append("get_job_details")
        else:
            parts.append("search_jobs")
        
        # Add extracted fields
        if extracted.get("category"):
            parts.append(f"category:{extracted['category']}")
        
        if extracted.get("location"):
            parts.append(f"location:{extracted['location']}")
        
        if extracted.get("salary_min"):
            parts.append(f"salary_min:{extracted['salary_min']}")
        
        if extracted.get("level"):
            parts.append(f"level:{extracted['level']}")
        
        if extracted.get("employment_type"):
            parts.append(f"employment_type:{extracted['employment_type']}")
        
        if extracted.get("job_id"):
            parts.append(f"job_id:{extracted['job_id']}")
        
        if extracted.get("keywords"):
            parts.append(f"keywords:{' '.join(extracted['keywords'][:3])}")
        
        return " ".join(parts)
    
    @staticmethod
    def suggest_corrections(query: str) -> Optional[str]:
        """Suggest corrections for common mistakes"""
        original = query.lower().strip()
        
        # Common typos
        typo_fixes = {
            "pytho": "python",
            "javscript": "javascript",
            "typscript": "typescript",
            "reac": "react",
            "angula": "angular",
            "hanoi": "hà nội",
            "hcm": "tp.hcm",
        }
        
        corrected = original
        changed = False
        for typo, fix in typo_fixes.items():
            if typo in corrected:
                corrected = corrected.replace(typo, fix)
                changed = True
        
        return corrected if changed else None


def normalize_query(query: str) -> str:
    """Convenience function to normalize query"""
    result = QueryNormalizer.normalize(query)
    return result["standardized"]


# Example usage:
if __name__ == "__main__":
    test_queries = [
        "tìm backend ở hà nội lương 25m",
        "be dev remote sr",
        "search react jobs in hcm",
        "chi tiết công việc id 6283",
        "frontend react vue angular ở tp.hcm junior fulltime",
        "pytho dev ha noi 20k",
    ]
    
    for query in test_queries:
        result = QueryNormalizer.normalize(query)
        print(f"Original:    {result['original']}")
        print(f"Normalized:  {result['normalized']}")
        print(f"Standardized: {result['standardized']}")
        print(f"Extracted:   {result['extracted']}")
        print()
