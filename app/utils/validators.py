import re
import logging

def _to_int_job_id(x):
    """Chuyển job_id về int an toàn (nhận int, '716', 'job_716'...)."""
    if isinstance(x, int):
        return x
    if isinstance(x, str):
        m = re.search(r"\d+", x)
        if m:
            try:
                return int(m.group())
            except Exception:
                return None
    return None

# Other validators if needed (e.g., email_validator)
def validate_email(email: str) -> bool:
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_phone(phone: str) -> bool:
    """Validate phone number (simple, supports Vietnamese +84)."""
    pattern = r'^\+?84|0\d{9,10}$'
    return re.match(pattern, phone.replace(' ', '').replace('-', '')) is not None