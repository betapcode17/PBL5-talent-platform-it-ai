from dateutil.parser import parse as _dt_parse
from datetime import datetime
import re
import logging

def normalize_date(date_str: str) -> str:
    """Chuẩn hóa định dạng ngày thành YYYY-MM-DD hoặc giữ 'Present'."""
    if not date_str or date_str.lower() == "present":
        return "Present"
    try:
        parsed = _dt_parse(date_str, fuzzy=True)
        return parsed.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        logging.warning(f"Invalid date format: {date_str}, defaulting to empty")
        return ""

def normalize_deadline(val: str) -> str:
    """Đưa deadline bất kỳ (vd 09/10/2025, 2025/10/09, 09-10-2025, '04/10/2025', ...) về YYYY-MM-DD.
    Không parse được thì trả về chuỗi rỗng."""
    if not val:
        return ""
    s = str(val).strip()
    if not s or s.lower() in {"n/a", "none", "null", "không xác định"}:
        return ""
    # Thử parse "dayfirst" để nhận dạng định dạng Việt Nam dd/mm/yyyy
    try:
        dt = _dt_parse(s, dayfirst=True, fuzzy=True)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        pass
    # Thử vài regex phổ biến nếu cần
    m = re.match(r"^(\d{2})[/-](\d{2})[/-](\d{4})$", s)
    if m:
        d, mth, y = m.groups()
        try:
            dt = datetime(int(y), int(mth), int(d))
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return ""
    return ""