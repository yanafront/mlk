import json
import os
import re
from typing import Any, Dict

import requests

VACANCY_NORMALIZER_API_URL = os.getenv("VACANCY_NORMALIZER_API_URL", "").strip()
VACANCY_NORMALIZER_API_TIMEOUT_SECONDS = int(
    os.getenv("VACANCY_NORMALIZER_API_TIMEOUT_SECONDS", "300")
)


def _extract_json_from_text(result: str) -> Dict[str, Any]:
    # 1. Ищем блок ```json ... ```
    json_blocks = re.findall(r"```json\s*(.*?)\s*```", result, re.DOTALL)
    if json_blocks:
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
        # 2. Берем последний полный JSON-объект
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

    # Удаляем комментарии
    json_str = re.sub(r",\s*//[^\n]*", ",", json_str)
    json_str = re.sub(r"(?m)^\s*//[^\n]*\n?", "", json_str)
    json_str = re.sub(r"/\*.*?\*/", "", json_str, flags=re.DOTALL)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        json_str_clean = re.sub(r",\s*$", "", json_str.strip())
        try:
            data = json.loads("[" + json_str_clean + "]")
            data = data[0] if isinstance(data, list) and data else data
        except json.JSONDecodeError:
            return {"error": "JSON parse error", "raw": result}

    if not isinstance(data, dict):
        return {"error": "empty or invalid result", "raw": result}

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
                            if isinstance(first_data, dict) and any(
                                v for v in first_data.values() if v
                            ):
                                return first_data
                        except json.JSONDecodeError:
                            pass
                        break

    return data


def normalize_vacancy_llm(vacancy_text: str) -> dict:
    """Нормализует вакансию через внешний API."""
    if not VACANCY_NORMALIZER_API_URL:
        return {"error": "VACANCY_NORMALIZER_API_URL is not set"}

    payload = {
        "vacancy_text": vacancy_text.strip(),
    }

    try:
        response = requests.post(
            VACANCY_NORMALIZER_API_URL,
            json=payload,
            timeout=VACANCY_NORMALIZER_API_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        return {"error": "vacancy normalizer API request failed", "details": str(exc)}

    try:
        response_data = response.json()
    except ValueError:
        return _extract_json_from_text(response.text)

    if isinstance(response_data, dict):
        if "normalized" in response_data:
            normalized = response_data["normalized"]
            if isinstance(normalized, dict):
                return normalized
            if isinstance(normalized, str):
                return _extract_json_from_text(normalized)

        if "result" in response_data:
            result = response_data["result"]
            if isinstance(result, dict):
                return result
            if isinstance(result, str):
                return _extract_json_from_text(result)

        return response_data

    if isinstance(response_data, str):
        return _extract_json_from_text(response_data)

    return {"error": "unexpected API response format", "raw": response_data}


def has_usable_normalized_vacancy_data(data) -> bool:
    """
    True, если из ответа нормализации можно строить поиск (есть хотя бы одно непустое поле).
    Ошибка API, {} или объект только из пустых полей — False.
    """
    if not isinstance(data, dict):
        return False
    if "error" in data:
        return False
    if not data:
        return False
    for key, value in data.items():
        if key == "skills":
            if isinstance(value, list) and any(str(x).strip() for x in value):
                return True
            continue
        if value not in (None, "", []):
            return True
    return False


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
    return "\n".join(
        [
            f"Job title: {data.get('job_title', '')}",
            f"Occupation: {data.get('occupation', '')}",
            f"Skills: {skills_str}",
            f"Work type: {data.get('work_type', '')}",
            f"Seniority: {data.get('seniority', '')}",
            f"Contact info: {data.get('contact_info', '')}",
            f"Location: {data.get('location', '')}",
            f"Salary: {data.get('salary', '')}",
            f"Employment type: {data.get('employment_type', '')}",
        ]
    )