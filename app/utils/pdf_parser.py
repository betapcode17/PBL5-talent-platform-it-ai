# app/utils/pdf_parser.py

import os
import re
import json
import logging
from typing import Dict, List, Any
from fastapi import HTTPException

from app.services.llm_service import get_llm_service


# =========================
# CONFIG
# =========================

SKILL_ALIASES = {
    "restful api design": "REST API",
    "rest api": "REST API",
    "rest apis": "REST API",
    "oauth 2.0": "OAuth2",
    "oauth2": "OAuth2",
    "jwt authentication": "JWT",
    "react.js": "ReactJS",
    "react": "ReactJS",
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "js": "JavaScript",
}

SKILL_CATEGORIES = {
    "backend": {
        "Java",
        "Spring Boot",
        "PHP",
        "Laravel",
        "REST API",
        "JWT",
        "OAuth2",
        "NodeJS",
        "ExpressJS",
    },
    "frontend": {
        "ReactJS",
        "JavaScript",
        "HTML5",
        "CSS3",
        "TypeScript",
    },
    "database": {
        "MySQL",
        "PostgreSQL",
        "MongoDB",
        "Redis",
    },
    "devops": {
        "Docker",
        "Kubernetes",
        "CI/CD",
        "AWS",
        "Nginx",
    },
    "tools": {
        "Git",
        "GitHub",
        "Postman",
        "IntelliJ IDEA",
    }
}


SECTION_HEADERS = {
    "skills": [
        "technical skills",
        "skills",
        "technologies",
    ],
    "projects": [
        "projects",
        "personal projects",
    ],
    "experience": [
        "experience",
        "work experience",
        "employment",
    ],
    "education": [
        "education",
        "academic background",
    ],
    "languages": [
        "languages",
    ]
}


# =========================
# PDF EXTRACTION
# =========================

def extract_text_from_pdf(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=400, detail="PDF file not found")

    if os.path.getsize(pdf_path) == 0:
        raise HTTPException(status_code=400, detail="PDF file is empty")

    try:
        import pdfplumber

        text = ""

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="No text extracted. PDF may be scanned image."
            )

        return text

    except Exception as e:
        logging.error(f"PDF extraction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract PDF text: {str(e)}"
        )


# =========================
# HELPERS
# =========================

def normalize_skill(skill: str) -> str:
    skill = skill.strip()
    skill = re.sub(r"\s+", " ", skill)

    lower = skill.lower()

    if lower in SKILL_ALIASES:
        return SKILL_ALIASES[lower]

    return skill


def deduplicate_skills(skills: List[str]) -> List[str]:
    seen = set()
    result = []

    for skill in skills:
        normalized = normalize_skill(skill)

        if normalized.lower() not in seen:
            seen.add(normalized.lower())
            result.append(normalized)

    return result


def extract_email(text: str) -> str:
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)

    return match.group(0) if match else ""


def extract_phone(text: str) -> str:
    match = re.search(r'(?:\+84|0)[0-9\s\-]{8,15}', text)

    return match.group(0).strip() if match else ""


def extract_name(text: str) -> str:
    lines = [x.strip() for x in text.split("\n") if x.strip()]

    for line in lines[:5]:
        if (
            2 <= len(line.split()) <= 5
            and not any(ch.isdigit() for ch in line)
            and re.search(r"[A-Za-zÀ-ỹ]", line)
        ):
            return line

    return ""


def split_sections(cv_text: str) -> Dict[str, str]:
    sections = {}

    lines = cv_text.split("\n")

    current_section = "general"
    buffer = []

    def save_section():
        nonlocal buffer

        if buffer:
            sections[current_section] = "\n".join(buffer).strip()

        buffer = []

    for line in lines:
        clean = line.strip()

        lower = clean.lower()

        matched = False

        for key, headers in SECTION_HEADERS.items():
            if lower in headers:
                save_section()
                current_section = key
                matched = True
                break

        if not matched:
            buffer.append(line)

    save_section()

    return sections


# =========================
# SKILL EXTRACTION
# =========================

def extract_skills(skills_text: str) -> List[str]:
    if not skills_text:
        return []

    raw_tokens = []

    lines = skills_text.split("\n")

    for line in lines:
        line = re.sub(
            r'^(Languages|Backend Concepts|Web Tech|Databases|Tools)\s*:',
            '',
            line,
            flags=re.IGNORECASE
        )

        parts = re.split(r'[•;,]', line)

        for part in parts:
            token = part.strip()

            if not token:
                continue

            # Handle Authentication (OAuth2, JWT)
            match = re.match(r'(.+?)\((.+?)\)', token)

            if match:
                base = match.group(1).strip()
                inner = match.group(2).strip()

                raw_tokens.append(base)

                inner_parts = [x.strip() for x in inner.split(",")]

                raw_tokens.extend(inner_parts)

            else:
                raw_tokens.append(token)

    clean = []

    for token in raw_tokens:
        token = token.strip().rstrip(".")

        if len(token) < 2:
            continue

        clean.append(normalize_skill(token))

    return deduplicate_skills(clean)


def categorize_skills(skills: List[str]) -> Dict[str, List[str]]:
    result = {
        "backend": [],
        "frontend": [],
        "database": [],
        "devops": [],
        "tools": [],
        "other": []
    }

    for skill in skills:
        matched = False

        for category, values in SKILL_CATEGORIES.items():
            if skill in values:
                result[category].append(skill)
                matched = True
                break

        if not matched:
            result["other"].append(skill)

    return result


# =========================
# PROJECT EXTRACTION
# =========================

def extract_projects(project_text: str) -> List[Dict[str, Any]]:
    if not project_text:
        return []

    projects = []

    blocks = re.split(r'\n(?=[A-Z])', project_text)

    for block in blocks:
        block = block.strip()

        if len(block) < 10:
            continue

        lines = block.split("\n")

        title = lines[0].strip()

        description = "\n".join(lines[1:]).strip()

        tech_stack = []

        stack_match = re.search(
            r'Tech Stack\s*:\s*(.+)',
            block,
            flags=re.IGNORECASE
        )

        if stack_match:
            tech_stack = extract_skills(stack_match.group(1))

        projects.append({
            "name": title,
            "description": description[:1000],
            "tech_stack": tech_stack,
        })

    return projects


# =========================
# EXPERIENCE EXTRACTION
# =========================

def extract_experience(exp_text: str) -> List[Dict[str, Any]]:
    if not exp_text:
        return []

    experiences = []

    blocks = re.split(r'\n(?=[A-Z])', exp_text)

    for block in blocks:
        block = block.strip()

        if len(block) < 10:
            continue

        lines = block.split("\n")

        title = lines[0].strip()

        description = "\n".join(lines[1:]).strip()

        date_match = re.search(
            r'(\d{4}|\d{2}/\d{4}).*?(Present|\d{4}|\d{2}/\d{4})?',
            block,
            flags=re.IGNORECASE
        )

        start_date = ""
        end_date = ""

        if date_match:
            start_date = date_match.group(1)

            if date_match.group(2):
                end_date = date_match.group(2)

        experiences.append({
            "company": "Unknown",
            "title": title,
            "start_date": start_date,
            "end_date": end_date,
            "description": description[:1000],
        })

    return experiences


# =========================
# EDUCATION EXTRACTION
# =========================

def extract_education(edu_text: str) -> List[Dict[str, Any]]:
    if not edu_text:
        return []

    result = []

    lines = [x.strip() for x in edu_text.split("\n") if x.strip()]

    for line in lines:
        if len(line) < 5:
            continue

        result.append({
            "school": line,
            "degree": "Unknown",
            "major": "Unknown",
            "start_date": "",
            "end_date": "",
        })

    return result


# =========================
# CAREER OBJECTIVE
# =========================

def extract_career_objective(cv_text: str) -> str:
    patterns = [
        r'(Aspiring.+?)(?:\n\n|\n[A-Z])',
        r'(Seeking.+?)(?:\n\n|\n[A-Z])',
        r'(Summary.+?)(?:\n\n|\n[A-Z])',
        r'(Objective.+?)(?:\n\n|\n[A-Z])',
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            cv_text,
            flags=re.IGNORECASE | re.DOTALL
        )

        if match:
            return match.group(1).strip()

    return ""


# =========================
# MAIN EXTRACTION
# =========================

def extract_cv_info_fallback(cv_text: str) -> dict:
    sections = split_sections(cv_text)

    skills = extract_skills(
        sections.get("skills", "")
    )

    categorized_skills = categorize_skills(skills)

    projects = extract_projects(
        sections.get("projects", "")
    )

    experience = extract_experience(
        sections.get("experience", "")
    )

    education = extract_education(
        sections.get("education", "")
    )

    result = {
        "name": extract_name(cv_text),
        "email": extract_email(cv_text),
        "phone": extract_phone(cv_text),
        "career_objective": extract_career_objective(cv_text),

        "skills": skills,

        "skill_categories": categorized_skills,

        "projects": projects,

        "experience": experience,

        "education": education,

        "has_projects": len(projects) > 0,

        "confidence": {
            "name": 0.9,
            "email": 1.0,
            "phone": 0.95,
            "skills": 0.85,
            "projects": 0.8,
            "experience": 0.75,
        }
    }

    return result


# =========================
# MAIN ENTRY
# =========================

def extract_cv_info(cv_text: str) -> dict:
    if not cv_text.strip():
        raise HTTPException(
            status_code=400,
            detail="CV text empty"
        )

    if len(cv_text.strip()) < 50:
        raise HTTPException(
            status_code=400,
            detail="PDF may be scanned image / OCR failed"
        )

    try:
        llm = get_llm_service()

        prompt = f"""
You are an ATS resume parser.

STRICT RULES:
- NEVER hallucinate
- ONLY use CV data
- Return VALID JSON ONLY
- If missing => empty array/string
- Preserve original language

CV:
\"\"\"
{cv_text}
\"\"\"
"""

        response = llm.generate_response(prompt)

        cleaned = re.sub(r"```json|```", "", response).strip()

        parsed = json.loads(cleaned)

        return parsed

    except Exception as e:
        logging.warning(
            f"LLM extraction failed: {str(e)}"
        )

        return extract_cv_info_fallback(cv_text)


def parse_cv_input_string(cv_text: str) -> dict:
    """Backward-compatible wrapper for older callers expecting the legacy parser name."""
    return extract_cv_info(cv_text)