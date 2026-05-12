"""Response formatting helpers for stable chatbot output."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


FIELD_ORDER = ["summary", "skills", "salary", "location", "nextstep"]
FIELD_LABELS = {
    "summary": "Summary:",
    "skills": "Skills:",
    "salary": "Salary:",
    "location": "Location:",
    "nextstep": "NextStep:",
}


def build_structured_chat_response(answer: str, sources: List[Dict[str, Any]]) -> Dict[str, str]:
    parsed = _parse_structured_text(answer or "")
    if parsed and not _looks_noisy_answer(parsed.get("summary", "")):
        return {
            "summary": _normalize_text_value(parsed.get("summary")),
            "skills": _normalize_text_value(parsed.get("skills")),
            "salary": _normalize_salary(parsed.get("salary")),
            "location": _normalize_text_value(parsed.get("location")),
            "nextStep": _normalize_text_value(parsed.get("nextStep")),
        }

    primary = sources[0] if sources else {}
    return {
        "summary": _fallback_summary(primary),
        "skills": _join_value(primary.get("skills")),
        "salary": _normalize_salary(primary.get("salary")),
        "location": _normalize_text_value(primary.get("location")),
        "nextStep": _suggest_next_step(primary),
    }


def format_chat_response(answer: str, sources: List[Dict[str, Any]]) -> str:
    """Render a short plain-text response for the chat UI."""
    primary = sources[0] if sources else {}
    parsed = _parse_structured_text(answer or "")
    normalized_answer = (answer or "").strip()

    if parsed and not _looks_noisy_answer(parsed.get("summary", "")):
        summary = _normalize_text_value(parsed.get("summary"))
        next_step = _normalize_text_value(parsed.get("nextStep"))
    else:
        summary = _fallback_summary(primary) if _looks_noisy_answer(normalized_answer) else _normalize_text_value(normalized_answer)
        if not summary or summary == "Chua co thong tin ro rang":
            summary = _fallback_summary(primary)
        next_step = _suggest_next_step(primary)

    if summary.endswith("."):
        return f"{summary} {next_step}".strip()
    return f"{summary}. {next_step}".strip()


def _parse_structured_text(text: str) -> Optional[Dict[str, str]]:
    lines = [line.rstrip() for line in (text or "").splitlines()]
    values: Dict[str, List[str]] = {}
    current_key: Optional[str] = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if current_key and values.get(current_key):
                values[current_key].append("")
            continue

        matched_key = _match_field_key(line)
        if matched_key:
            if matched_key in values:
                break
            current_key = matched_key
            values[current_key] = [_extract_inline_value(line, FIELD_LABELS[matched_key])]
            continue

        if current_key:
            values.setdefault(current_key, []).append(line)

    if not all(key in values for key in FIELD_ORDER):
        return None

    result = {
        "summary": _collapse_lines(values["summary"]),
        "skills": _collapse_lines(values["skills"]),
        "salary": _collapse_lines(values["salary"]),
        "location": _collapse_lines(values["location"]),
        "nextStep": _collapse_lines(values["nextstep"]),
    }
    return result


def _match_field_key(line: str) -> Optional[str]:
    lowered = line.lower()
    for key, label in FIELD_LABELS.items():
        if lowered.startswith(label.lower()):
            return key
    return None


def _extract_inline_value(line: str, label: str) -> str:
    return line[len(label):].strip() if line.lower().startswith(label.lower()) else ""


def _collapse_lines(lines: List[str]) -> str:
    cleaned = [segment.strip() for segment in lines if segment is not None]
    text = "\n".join(segment for segment in cleaned if segment)
    return _normalize_text_value(text)


def _join_value(value: Optional[Any]) -> str:
    if value is None:
        return "Chua co thong tin ro rang"
    if isinstance(value, list):
        cleaned = [_normalize_text_value(item) for item in value]
        cleaned = [item for item in cleaned if item and item != "Chua co thong tin ro rang"]
        return ", ".join(cleaned) if cleaned else "Chua co thong tin ro rang"
    return _normalize_text_value(value)


def _normalize_salary(value: Optional[Any]) -> str:
    if isinstance(value, dict):
        minimum = value.get("min")
        maximum = value.get("max")
        currency = value.get("currency")
        if minimum and maximum:
            suffix = f" {currency}" if currency else ""
            return f"{minimum} - {maximum}{suffix}".strip()
        if minimum:
            suffix = f" {currency}" if currency else ""
            return f"Tu {minimum}{suffix}".strip()
        if maximum:
            suffix = f" {currency}" if currency else ""
            return f"Den {maximum}{suffix}".strip()
        return "Chua co thong tin ro rang"

    text = _normalize_text_value(value)
    if text in {"{'min': None, 'max': None}", "{'min': none, 'max': none}"}:
        return "Chua co thong tin ro rang"
    return text


def _normalize_text_value(value: Optional[Any]) -> str:
    if value is None:
        return "Chua co thong tin ro rang"

    text = str(value).strip()
    if not text:
        return "Chua co thong tin ro rang"

    lowered = text.lower()
    noisy_values = {
        "khong co du lieu",
        "khong co thong tin",
        "khong ro",
        "khong xac dinh",
        "chua co thong tin",
        "chua co thong tin ro rang",
        "none",
        "null",
        "n/a",
    }
    if lowered in noisy_values:
        return "Chua co thong tin ro rang"

    return text


def _looks_noisy_answer(text: str) -> bool:
    lowered = text.lower()
    noisy_tokens = [
        "{'company_id'",
        '"company_id"',
        "tool",
        "response to clarify",
        "provide example",
        "no relevant backend context found",
    ]
    return any(token in lowered for token in noisy_tokens)


def _fallback_summary(primary: Dict[str, Any]) -> str:
    title = _normalize_text_value(primary.get("title"))
    company = _normalize_text_value(primary.get("company"))
    if title != "Chua co thong tin ro rang" and company != "Chua co thong tin ro rang":
        return f"Vi tri phu hop nhat hien tai la {title} tai {company}."
    if title != "Chua co thong tin ro rang":
        return f"Vi tri phu hop nhat hien tai la {title}."
    return "Chua tim thay du lieu du manh de tom tat ro rang."


def _suggest_next_step(primary: Dict[str, Any]) -> str:
    title = _normalize_text_value(primary.get("title"))
    if title != "Chua co thong tin ro rang":
        return f"Neu ban quan tam, hay xem ky mo ta va yeu cau cua vi tri {title} de danh gia muc do phu hop."
    return "Hay cung cap them yeu cau cu the nhu ky nang, muc luong hoac dia diem de tim ket qua sat hon."
