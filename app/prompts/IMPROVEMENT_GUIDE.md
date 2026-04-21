# app/prompts/IMPROVEMENT_GUIDE.md

# 🚀 Prompt & Context Improvements Guide

## ✨ Tóm Tắt Các Cải Tiến

Dự án đã được nâng cấp với:

- ✅ **Enhanced System Prompts** - Prompt rõ ràng, có cấu trúc
- ✅ **Context Builders** - Xây dựng context một cách chuẩn hóa
- ✅ **Prompt Engineering Utilities** - Tools cho prompt engineering best practices
- ✅ **Output Format Templates** - Templates định dạng output
- ✅ **Quality Checklist** - Kiểm tra chất lượng prompt/response

---

## 📚 Các Files Mới

```
app/prompts/
├── chatbot_system_prompt.py      [UPGRADED] - Enhanced system prompts
├── context_builder.py             [NEW] - Xây dựng context
├── prompt_engineering.py           [NEW] - Prompt engineering utilities
├── __init__.py                     [UPDATED] - Export tất cả functions
└── IMPROVEMENT_GUIDE.md            [THIS FILE]
```

---

## 🎯 Cách Sử Dụng Trong chatbot_service.py

### Option 1: Sử dụng Context Builders

**Trước:**

```python
# chatbot_service.py
context = "\n---\n".join(context_parts)
full_prompt = f"{conv_context}\n\n=== DU LIEU ===\n{context}\n\nCau hoi: {user_message}"
```

**Sau:**

```python
from app.prompts import build_job_search_context, build_constrained_prompt
from app.prompts import get_system_prompt, get_data_constraints

# Xây dựng context rõ ràng
context = build_job_search_context(
    user_profile=user_profile_dict,
    market_data=market_stats,
    retrieved_jobs=job_results,
    conversation_history=conv_history,
    user_message=user_message
)

# Build prompt rõ ràng với constraints
system_prompt = get_system_prompt("jobs")
constraints = get_data_constraints()
full_prompt = build_constrained_prompt(
    user_message=user_message,
    context=context,
    system_role=system_prompt,
    constraints=constraints
)

response = llm.generate(full_prompt)
```

### Option 2: Sử dụng Chain-of-Thought (CoT)

**Thêm reasoning step cho LLM:**

```python
from app.prompts import build_prompt_with_cot, get_cot_prompt

cot = get_cot_prompt("matching")
prompt = build_prompt_with_cot(
    user_message=user_message,
    context=context,
    system_role=system_prompt,
    cot_instruction=cot
)

response = llm.generate(prompt)
```

### Option 3: Output Formatting

**Định dạng output rõ ràng:**

```python
from app.prompts import create_structured_output_template

output_format = create_structured_output_template("ranked_list")

prompt = f"""
{system_prompt}

CONTEXT:
{context}

OUTPUT FORMAT:
{output_format}

QUESTION:
{user_message}

RESPONSE:
"""
```

---

## 📝 Integration Example - Chatbot Service Update

```python
# app/services/chatbot_service.py

from app.prompts import (
    build_job_search_context,
    build_constrained_prompt,
    get_system_prompt,
    get_data_constraints,
    get_quality_constraints,
    format_conversation_for_context,
)

class ChatbotRAG:
    def chat(self, user_message, session_id=None, context_type="auto"):
        """Enhanced chat with better context and prompts"""

        # ... existing code ...

        # 1. Build comprehensive context
        context = build_job_search_context(
            user_profile={
                'level': 'Senior',
                'experience_years': 5,
                'skills': ['Python', 'Django', 'FastAPI'],
                'location': 'TP.HCM',
                'work_type': 'remote',
            },
            market_data=self.retrieval_service.get_collection_stats(),
            retrieved_jobs=doc_results,
            conversation_history=self.get_history(session_id, limit=3),
            user_message=user_message
        )

        # 2. Get system prompt and constraints
        system_prompt = get_system_prompt(detected_intent)
        constraints = get_data_constraints() + get_quality_constraints()

        # 3. Build enhanced prompt
        full_prompt = build_constrained_prompt(
            user_message=user_message,
            context=context,
            system_role=system_prompt,
            constraints=constraints,
            format_instruction="Dùng bullets, clear structure, explain WHY"
        )

        # 4. Generate response
        bot_response = self.llm_service.generate_response(full_prompt)

        # ... existing code ...
```

---

## 🔧 Các Functions Quan Trọng

### Context Builders

```python
# Để build context cho các intent khác nhau
build_job_search_context(...)      # Cho "jobs" intent
build_cv_analysis_context(...)     # Cho "cv" intent
build_matching_context(...)        # Cho "matching" intent
build_career_advice_context(...)   # Cho "career" intent
```

### Prompt Building

```python
# Để xây dựng prompt với best practices
build_prompt_with_cot(...)         # Chain-of-Thought
build_few_shot_prompt(...)         # Few-shot examples
build_constrained_prompt(...)      # Có constraints
build_comparison_prompt(...)       # Compare items
```

### Quality & Validation

```python
# Để check chất lượng
check_prompt_quality(prompt)           # Returns dict
calculate_prompt_score(checks)         # Returns 0-1 score
suggest_prompt_improvements(checks)    # Returns list
validate_context(context)              # Checks min length
```

---

## 💡 Best Practices

### ✅ DO's

1. **Luôn use structured context**

   ```python
   context = build_job_search_context(...)
   ```

2. **Luôn add constraints**

   ```python
   constraints = get_data_constraints()
   ```

3. **Luôn specify output format**

   ```python
   format_template = create_structured_output_template("ranked_list")
   ```

4. **Validate context before using**

   ```python
   if validate_context(context):
       # use context
   ```

5. **Check LLM quality**
   ```python
   checks = check_prompt_quality(prompt)
   score = calculate_prompt_score(checks)
   ```

### ❌ DON'Ts

1. ❌ **Don't use hardcoded prompts**
   - Dùng `get_system_prompt()` thay vì hardcode

2. ❌ **Don't skip constraints**
   - Luôn add data/quality constraints

3. ❌ **Don't ignore output format**
   - Specify exactly how LLM should format output

4. ❌ **Don't use empty context**
   - Validate context before using

---

## 📊 Output Format Templates Available

```
"ranked_list"      - Ranking items (jobs, CVs)
"analysis"         - Analysis with strengths/concerns
"matching"         - Match score with details
"comparison"       - Comparison matrix
"statistics"       - Market statistics
"roadmap"          - Development roadmap/timeline
```

**Example:**

```python
template = create_structured_output_template("matching")
# Returns:
# **Match Analysis: [Score]%**
# ✅ **Match Points ([X] of [Y]):**
# ...
```

---

## 🎯 Quick Start Example

```python
# 1. Import
from app.prompts import (
    build_job_search_context,
    build_constrained_prompt,
    get_system_prompt,
    get_data_constraints,
)

# 2. Build context
context = build_job_search_context(
    user_profile=user_data,
    market_data=market_stats,
    retrieved_jobs=jobs,
    user_message=msg
)

# 3. Build prompt
prompt = build_constrained_prompt(
    user_message=msg,
    context=context,
    system_role=get_system_prompt("jobs"),
    constraints=get_data_constraints()
)

# 4. Generate
response = llm.generate(prompt)
```

---

## 📋 Checklist for Better Prompts

Trước khi generate response, check:

- [ ] Context có role rõ ràng?
- [ ] Context có background info?
- [ ] Context có data thực tế?
- [ ] Context có instructions?
- [ ] Prompt có constraints?
- [ ] Output format specified?
- [ ] Context length valid? (> 200 chars)
- [ ] LLM score > 0.7?

---

## 🚀 Next Steps

1. **Update chatbot_service.py** - Use new context builders
2. **Update routers** - Pass structured data to chatbot
3. **Test quality** - Use check_prompt_quality()
4. **Monitor output** - Check response format consistency
5. **Iterate** - Improve based on quality metrics

---

## ❓ FAQ

**Q: Có bắt buộc use context builders không?**  
A: Không bắt buộc nhưng STRONGLY RECOMMENDED. Nó giúp context rõ ràng, chuẩn hóa, và dễ maintain.

**Q: Cách tùy chỉnh output format?**  
A: Modify `OUTPUT_FORMAT_EXAMPLES` trong `chatbot_system_prompt.py` hoặc tạo template custom.

**Q: Constraint nào là quan trọng nhất?**  
A: `get_data_constraints()` - để tránh LLM tự bịa data.

**Q: Làm sao biết prompt tốt hay không?**  
A: Dùng `check_prompt_quality()` và `calculate_prompt_score()`.

---

## 📞 Support

Nếu có thắc mắc:

1. Check example code ở file này
2. Refer docstrings trong từng file
3. Look at functions in `__init__.py`
