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
    do_sample=False,
    return_full_text=False,  # только сгенерированный текст, без промпта
)

# ---------- PROMPT TEMPLATE ----------
PROMPT_TEMPLATE = """
Извлеки структурированные данные из вакансии.
Верни ТОЛЬКО один JSON-объект, без объяснений, без текста до и после.
Схема:
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
- location = местоположение (город, страна). Нельзя угадывать страну, если нет в тексте.
- Ответ: строго один JSON, ничего больше. Не добавляй пояснений, извинений и второго пустого JSON.

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
    print("LLM response:", result)

    # ---------- EXTRACT JSON ----------
    # 1. Ищем блок ```json ... ``` — модель часто оборачивает ответ в markdown
    json_blocks = re.findall(r"```json\s*(.*?)\s*```", result, re.DOTALL)
    if json_blocks:
        # Модель иногда сначала даёт валидный JSON, потом извиняется и добавляет пустой {}
        # Берём непустой блок с данными, а не последний
        candidates = [b.strip() for b in json_blocks]
        json_str = None
        for c in reversed(candidates):
            try:
                parsed = json.loads(c)
                if isinstance(parsed, dict) and any(v for v in parsed.values() if v):
                    json_str = c
                    break
            except json.JSONDecodeError:
                pass
        json_str = json_str or candidates[-1]
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

    # Удаляем комментарии — JSON их не поддерживает, модель иногда их добавляет
    # Только после запятой (,\s*//) или целые строки-комментарии, чтобы не задеть // внутри строк
    json_str = re.sub(r",\s*//[^\n]*", ",", json_str)
    json_str = re.sub(r"(?m)^\s*//[^\n]*\n?", "", json_str)
    json_str = re.sub(r"/\*.*?\*/", "", json_str, flags=re.DOTALL)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # Модель может вернуть несколько объектов через запятую: {...}, {...}
        json_str_clean = re.sub(r",\s*$", "", json_str.strip())
        try:
            data = json.loads("[" + json_str_clean + "]")
            data = data[0] if isinstance(data, list) and data else data
        except json.JSONDecodeError:
            return {"error": "JSON parse error", "raw": result}

    # Пустой массив или не dict — считаем ошибкой
    if not isinstance(data, dict):
        return {"error": "empty or invalid result", "raw": result}

    # Если получили пустой dict, но в ответе есть JSON с данными (модель могла добавить {} после извинений)
    if not any(v for v in data.values() if v):
        first_brace = result.find("{")
        if first_brace != -1:
            depth, json_start = 0, first_brace
            for i in range(first_brace, len(result)):
                if result[i] == "{":
                    depth += 1
                elif result[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            first_data = json.loads(result[json_start : i + 1])
                            if isinstance(first_data, dict) and any(v for v in first_data.values() if v):
                                return first_data
                        except json.JSONDecodeError:
                            pass
                        break

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
