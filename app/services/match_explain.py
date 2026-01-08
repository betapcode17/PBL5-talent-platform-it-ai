from typing import Dict, Any

def normalize_explanation(raw) -> Dict[str, Any]:
    """
    Normalize explanation từ LLM:
    - None -> empty structure
    - str  -> đưa vào skills.reason
    - dict -> fill missing fields
    """
    base = {
        "skills": {"matched": [], "reason": ""},
        "experience": {"matched": [], "reason": ""},
        "aspirations": {"matched": [], "reason": ""},
        "education": {"matched": [], "reason": ""},
    }

    if raw is None:
        return base

    # Nếu LLM trả về STRING → nhét vào skills.reason
    if isinstance(raw, str):
        base["skills"]["reason"] = raw
        return base

    # Nếu là dict
    if isinstance(raw, dict):
        for k in base.keys():
            v = raw.get(k)
            if isinstance(v, dict):
                base[k]["matched"] = v.get("matched", []) or []
                base[k]["reason"] = v.get("reason", "") or ""
            elif isinstance(v, str):
                base[k]["reason"] = v
        return base

    return base


def build_why_match(expl: Dict[str, Any]) -> str:
    """
    Build câu giải thích tiếng Việt gọn gàng cho FE
    """
    parts = []

    if expl["skills"]["matched"]:
        parts.append(
            f"Khớp kỹ năng: {', '.join(expl['skills']['matched'][:3])}"
        )

    if expl["experience"]["reason"]:
        parts.append(expl["experience"]["reason"])

    if expl["education"]["reason"]:
        parts.append(expl["education"]["reason"])

    if expl["aspirations"]["reason"]:
        parts.append(expl["aspirations"]["reason"])

    return " | ".join(parts) or "Phù hợp tổng thể với yêu cầu vị trí"
