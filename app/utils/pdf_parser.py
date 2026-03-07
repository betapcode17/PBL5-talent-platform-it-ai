import os
import pdfplumber
import google.generativeai as genai
import re
import json
from tenacity import retry, stop_after_attempt, wait_exponential
from fastapi import HTTPException
from config import GOOGLE_API_KEY
import logging

genai.configure(api_key=GOOGLE_API_KEY)

def extract_text_from_pdf(pdf_path: str) -> str:
    """Trích xuất văn bản từ file PDF."""
    if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) == 0:
        raise HTTPException(status_code=400, detail="PDF file is empty or does not exist")
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                content = page.extract_text()
                if content:
                    text += content + "\n"
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text extracted from PDF")
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to extract text from PDF: {str(e)}")

def extract_cv_info(cv_text: str) -> dict:
    """Trích xuất thông tin CV từ văn bản, trả về JSON theo schema."""
    if not cv_text.strip():
        raise HTTPException(status_code=400, detail="CV text is empty")
    prompt = f"""
    Extract key resume information from the following CV text.
    Return JSON with this exact schema:
    {{
      "name": "",
      "email": "",
      "phone": "",
      "career_objective": "",
      "skills": [],
      "education": [
        {{
          "school": "",
          "degree": "",
          "major": "",
          "start_date": "YYYY-MM-DD",
          "end_date": "YYYY-MM-DD"
        }}
      ],
      "experience": [
        {{
          "company": "non-empty string",
          "title": "",
          "start_date": "YYYY-MM-DD or Present",
          "end_date": "YYYY-MM-DD or Present",
          "description": ""
        }}
      ]
    }}
    IMPORTANT RULES:
    - PRESERVE THE ORIGINAL LANGUAGE of all text fields (name, company, title, description, skills, etc.)
    - DO NOT translate Vietnamese to English or vice versa
    - If the CV is in Vietnamese, keep all data in Vietnamese
    - If the CV is in English, keep all data in English
    - Dates must be in YYYY-MM-DD format (e.g., '2022-01-01') or 'Present' for ongoing experiences
    - The 'company' field must be a non-empty string (use 'Unknown' if not provided)
    CV Text:
    \"\"\"{cv_text}\"\"\"\n"""
    try:
        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
        def call_gemini():
            model = genai.GenerativeModel("gemini-2.0-flash-lite")
            return model.generate_content(prompt)
       
        response = call_gemini()
        result = response.text
        cleaned = re.sub(r"```json|```", "", result).strip()
        cv_info = json.loads(cleaned)
        # Đảm bảo dữ liệu hợp lệ
        from utils.date_utils import normalize_date
        for exp in cv_info.get("experience", []):
            exp["company"] = exp.get("company") or "Unknown"
            exp["title"] = exp.get("title") or "Unknown"
            exp["description"] = exp.get("description") or "No description provided"
            exp["start_date"] = normalize_date(exp.get("start_date", ""))
            exp["end_date"] = normalize_date(exp.get("end_date", ""))
        for edu in cv_info.get("education", []):
            edu["school"] = edu.get("school") or "Unknown"
            edu["degree"] = edu.get("degree") or "Unknown"
            edu["major"] = edu.get("major") or "Unknown"
            edu["start_date"] = normalize_date(edu.get("start_date", ""))
            edu["end_date"] = normalize_date(edu.get("end_date", ""))
        logging.info(f"Extracted CV info: {json.dumps(cv_info, ensure_ascii=False)[:500]}...")
        return cv_info
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing CV info JSON: {str(e)} - Response: {result[:100]}...")
        raise HTTPException(status_code=500, detail="Failed to parse CV information")
    except Exception as e:
        logging.error(f"Error extracting CV info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to extract CV information")

def parse_cv_input_string(cv_input: str) -> dict:
    """Parse chuỗi cv_input thành dictionary."""
    try:
        # Nếu cv_input là JSON string, parse trực tiếp
        if cv_input.strip().startswith('{'):
            return json.loads(cv_input)
        # Nếu cv_input là chuỗi text, parse thủ công
        cv_info = {
            "skills": [],
            "career_objective": "",
            "experience": [],
            "education": [],
            "name": "",
            "email": "",
            "phone": ""
        }
        # Giả định format: "Skills: ...; Aspirations: ...; Experience: ...; Education: ..."
        sections = re.split(r'(Skills|Aspirations|Experience|Education|Name|Email|Phone):', cv_input, flags=re.IGNORECASE)
        from utils.date_utils import normalize_date
        for i in range(1, len(sections), 2):
            key = sections[i].lower()
            value = sections[i + 1].strip()
            if key == "skills":
                cv_info["skills"] = [s.strip() for s in value.split(',') if s.strip()]
            elif key == "aspirations":
                cv_info["career_objective"] = value
            elif key == "name":
                cv_info["name"] = value
            elif key == "email":
                cv_info["email"] = value
            elif key == "phone":
                cv_info["phone"] = value
            elif key == "experience":
                exp_entries = value.split('\n')
                for entry in exp_entries:
                    if entry.strip():
                        exp = {"company": "Unknown", "title": "Unknown", "start_date": "", "end_date": "", "description": ""}
                        fields = re.split(r';|,', entry)
                        for field in fields:
                            if ':' in field:
                                k, v = field.split(':', 1)
                                k = k.strip().lower()
                                v = v.strip()
                                if k in ["company", "title", "description"]:
                                    exp[k] = v
                                elif k in ["start_date", "end_date"]:
                                    exp[k] = normalize_date(v)
                        cv_info["experience"].append(exp)
            elif key == "education":
                edu_entries = value.split('\n')
                for entry in edu_entries:
                    if entry.strip():
                        edu = {"school": "Unknown", "degree": "Unknown", "major": "Unknown", "start_date": "", "end_date": ""}
                        fields = re.split(r';|,', entry)
                        for field in fields:
                            if ':' in field:
                                k, v = field.split(':', 1)
                                k = k.strip().lower()
                                v = v.strip()
                                if k in ["school", "degree", "major"]:
                                    edu[k] = v
                                elif k in ["start_date", "end_date"]:
                                    edu[k] = normalize_date(v)
                        cv_info["education"].append(edu)
        return cv_info
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in cv_input string: {cv_input[:100]}...")
        raise HTTPException(status_code=400, detail="Invalid JSON format in cv_input")
    except Exception as e:
        logging.error(f"Error parsing cv_input string: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to parse cv_input string: {str(e)}")