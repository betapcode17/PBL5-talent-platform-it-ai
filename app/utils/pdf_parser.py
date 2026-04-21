import os
import pdfplumber
import re
import json
from tenacity import retry, stop_after_attempt, wait_exponential
from fastapi import HTTPException
import logging

from app.services.llm_service import get_llm_service

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

def extract_cv_info_fallback(cv_text: str) -> dict:
    """Fallback method: Extract CV info using regex (no LLM required)"""
    cv_info = {
        "name": "",
        "email": "",
        "phone": "",
        "career_objective": "",
        "skills": [],
        "education": [],
        "experience": []
    }
    
    try:
        from app.utils.date_utils import normalize_date
        
        # Extract name (first line or after header)
        lines = cv_text.split('\n')
        for line in lines[:5]:  # Check first 5 lines
            if len(line.strip()) > 2 and len(line.strip()) < 100 and not any(c.isdigit() for c in line[:20]):
                cv_info["name"] = line.strip()
                break
        
        # Extract email
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', cv_text)
        if email_match:
            cv_info["email"] = email_match.group()
        
        # Extract phone (Vietnamese format or international)
        phone_match = re.search(r'(?:\+84|0)[\d\s\-]{9,}', cv_text)
        if phone_match:
            cv_info["phone"] = phone_match.group().strip()
        
        # Extract skills (look for "Kỹ năng" or "Skills" section)
        skills_match = re.search(r'(?:kỹ\s*năng|skills)[:\s]+(.*?)(?:\n\n|(?:kinh|experience|education|học))', cv_text, re.IGNORECASE | re.DOTALL)
        if skills_match:
            skills_text = skills_match.group(1)
            # Split by comma, bullet, or newline
            skill_list = re.split(r'[,•\n]', skills_text)
            cv_info["skills"] = [s.strip() for s in skill_list if s.strip() and len(s.strip()) > 2][:10]
        
        # Extract education
        edu_matches = re.finditer(r'(?:đại\s*học|university|school|trường)[:\s]+(.*?)(?:\n|$)', cv_text, re.IGNORECASE)
        for match in edu_matches:
            edu_text = match.group(1).strip()
            if edu_text:
                cv_info["education"].append({
                    "school": edu_text[:60],
                    "degree": "Degree",
                    "major": "Major",
                    "start_date": "2016-09-01",
                    "end_date": "2020-06-30"
                })
        
        # Extract experience
        exp_matches = re.finditer(r'(?:kinh\s*nghiệm|experience)[:\s]+(.*?)(?:kinh|education|học|$)', cv_text, re.IGNORECASE | re.DOTALL)
        for match in exp_matches:
            exp_text = match.group(1).strip()
            exp_lines = exp_text.split('\n')[:3]  # Take first 3 lines
            for line in exp_lines:
                if line.strip() and len(line.strip()) > 5:
                    cv_info["experience"].append({
                        "company": "Company",
                        "title": line.strip()[:60],
                        "start_date": "2020-01-01",
                        "end_date": "Present",
                        "description": line.strip()
                    })
        
        logging.info(f"[FALLBACK] Extracted CV info (regex-based)")
        return cv_info
        
    except Exception as e:
        logging.error(f"[FALLBACK ERROR] {str(e)}")
        return cv_info


def extract_cv_info(cv_text: str) -> dict:
    """Trích xuất thông tin CV từ văn bản, trả về JSON theo schema."""
    if not cv_text.strip():
        raise HTTPException(status_code=400, detail="CV text is empty")
    
    # Try LLM first (Ollama) - but with short timeout
    try:
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
    - PRESERVE THE ORIGINAL LANGUAGE of all text fields
    - Dates must be in YYYY-MM-DD format or 'Present'
    - The 'company' field must be a non-empty string (use 'Unknown' if not provided)
    CV Text:
    \"\"\"{cv_text}\"\"\"\n"""
        
        llm = get_llm_service()
        logging.info(f"[CV EXTRACT] Calling Ollama for CV extraction...")
        response = llm.generate_response(prompt)
        
        logging.info(f"[CV EXTRACT] Ollama response received ({len(response)} chars)")
        cleaned = re.sub(r"```json|```", "", response).strip()
        cv_info = json.loads(cleaned)
        
        # Validate required fields
        from app.utils.date_utils import normalize_date
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
        
        logging.info(f"[CV EXTRACT] ✅ LLM extraction successful")
        return cv_info
        
    except Exception as e:
        logging.warning(f"[CV EXTRACT] ⚠️ LLM extraction failed ({type(e).__name__}: {str(e)[:80]}) - Using fallback")
        # Fallback: use regex-based extraction (NO LLM REQUIRED)
        cv_info = extract_cv_info_fallback(cv_text)
        return cv_info

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