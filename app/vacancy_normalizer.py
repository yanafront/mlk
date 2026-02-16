import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"   # можно заменить на 3B

# ---------- LOAD MODEL ----------
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.1,
    do_sample=False
)

# ---------- PROMPT TEMPLATE ----------
PROMPT_TEMPLATE = """
Извлеки структурированные данные из вакансии.
Верни JSON строго по схеме:
{{
  "job_title": "",
  "occupation": "",
  "skills": [],
  "work_type": "",
  "seniority": "",
  "contact_info": "",
  "location": "",
  "salary": "",
  "employment_type": ""
}}

Правила:
- occupation = категория профессии (IT, Продажи, Логистика, Физический труд и т.д.)
- Если данных нет → ""
- skills = список технологий или навыков
- salary = зарплата
- employment_type = тип занятости (полный день, неполный день, удаленная работа, гибрид)
- contact_info = контактная информация (email, телефон, skype, telegram)
- location = местоположение (город, страна)
- Верни только JSON без текста

Текст вакансии:
\"\"\"
{vacancy_text}
\"\"\"
"""

# ---------- NORMALIZATION FUNCTION ----------
def normalize_vacancy_llm(vacancy_text: str) -> dict:
    prompt = PROMPT_TEMPLATE.format(
        vacancy_text=vacancy_text.strip()
    )

    result = generator(prompt)[0]["generated_text"]

    # ---------- EXTRACT JSON ----------
    # 1. Ищем блок ```json ... ``` — модель часто оборачивает ответ в markdown
    json_blocks = re.findall(r"```json\s*(.*?)\s*```", result, re.DOTALL)
    if json_blocks:
        json_str = json_blocks[-1].strip()
    else:
        # 2. Берём последний полный JSON-объект (ответ модели в конце)
        last_brace = result.rfind("}")
        if last_brace == -1:
            return {"error": "JSON not found", "raw": result}
        depth = 0
        json_start = -1
        for i in range(last_brace, -1, -1):
            if result[i] == "}":
                depth += 1
            elif result[i] == "{":
                depth -= 1
                if depth == 0:
                    json_start = i
                    break
        if json_start == -1:
            return {"error": "JSON not found", "raw": result}
        json_str = result[json_start : last_brace + 1]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return {"error": "JSON parse error", "raw": result}

    return data


def normalized_data_to_embedding_text(data) -> str:
    """Формирует текст для embedding из нормализованных данных."""
    if not isinstance(data, dict):
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            data = data[0]
        else:
            return ""
    if "error" in data:
        return ""
    skills = data.get("skills") or []
    skills_str = ", ".join(skills) if isinstance(skills, list) else str(skills)
    return "\n".join([
        f"Job title: {data.get('job_title', '')}",
        f"Occupation: {data.get('occupation', '')}",
        f"Skills: {skills_str}",
        f"Work type: {data.get('work_type', '')}",
        f"Seniority: {data.get('seniority', '')}",
        f"Contact info: {data.get('contact_info', '')}",
        f"Location: {data.get('location', '')}",
        f"Salary: {data.get('salary', '')}",
        f"Employment type: {data.get('employment_type', '')}",
    ])
