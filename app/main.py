from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import math
from app.search import search_vacancies, search_vacancies_without_rerank, search_users_by_vacancy
from app.db import get_conn
from app.models import embedding_model, reranker_model
from app.schemas import (
    EmbedRequest,
    VacancyMatchRequest,
    AddUserRequest,
)

app = FastAPI(title="Job Semantic Search ML Service")


class SearchRequest(BaseModel):
    text: str
    top_n: int = 5 


def score_to_percent(score: float) -> int:
    return int(100 / (1 + math.exp(-score)))

def normalize_scores_to_percent(scores):
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)

    if max_s == min_s:
        return [50 for _ in scores]

    return [
        int(100 * (s - min_s) / (max_s - min_s))
        for s in scores
    ]


@app.post("/query")
def process_query(content: str):
    # сохраняем запрос
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO main (content) VALUES (%s)",
        (content,)
    )
    conn.commit()
    cur.close()
    conn.close()

    # ищем вакансии
    results = search_vacancies(content)
    return results

@app.post("/embed")
def embed(req: EmbedRequest):
    # E5: префикс "query: " для запросов
    text = f"query: {req.text}"
    embedding = embedding_model.encode(
        text,
        normalize_embeddings=True
    )
    return {"embedding": embedding.tolist()}


@app.post("/search")
def search(req: SearchRequest):
    top_n = max(1, min(req.top_n, 20))  # защита: 1..20

    results = search_vacancies(req.text)

    top_results = results[:top_n]

    scores = [r["score"] for r in top_results]
    percents = normalize_scores_to_percent(scores)

   
    response = []
    for r, p in zip(top_results[:5], percents):
        response.append({
            "vacancy_id": r["id"],
            "text": r["content"],
            "relevance_percent": p
        })

    return {
        "query": req.text,
        "results": response
    }



@app.post("/users")
def add_user(req: AddUserRequest):
    """
    Добавляет или обновляет пользователя (кандидата) с профилем/резюме.
    Если user_id уже есть — обновляем description и embedding, иначе создаём.
    """
    if not req.description or len(req.description.strip()) < 10:
        raise HTTPException(status_code=400, detail="content слишком короткий (минимум 10 символов)")

    embedding = embedding_model.encode(
        f"passage: {req.description.strip()}",
        normalize_embeddings=True
    ).tolist()

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id FROM main WHERE user_id = %s::bigint LIMIT 1",
        (req.user_id,)
    )
    row = cur.fetchone()
    if row:
        cur.execute(
            "UPDATE main SET description = %s, embedding = %s WHERE user_id = %s::bigint",
            (req.description.strip(), embedding, req.user_id)
        )
        conn.commit()
        cur.close()
        conn.close()
        return {"id": row[0], "status": "updated"}
    else:
        cur.execute(
            "INSERT INTO main (user_id, description, embedding) VALUES (%s::bigint, %s, %s) RETURNING id",
            (req.user_id, req.description.strip(), embedding)
        )
        user_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        return {"id": user_id, "status": "created"}


@app.post("/vacancy/match_users")
def match_users_by_vacancy(req: VacancyMatchRequest):
    """
    Принимает текст вакансии и возвращает подходящих пользователей (кандидатов).
    """
    top_n = max(1, min(req.top_n, 50))
    results = search_users_by_vacancy(req.vacancy_text, top_k=top_n)

    scores = [r["score"] for r in results]
    percents = normalize_scores_to_percent(scores) if scores else []

    response = []
    for r, p in zip(results, percents):
        response.append({
            "user_id": r["id"],
            "profile": r["description"],
            "relevance_percent": p
        })

    return {
        "vacancy_text": req.vacancy_text[:200] + ("..." if len(req.vacancy_text) > 200 else ""),
        "results": response
    }


@app.post("/search_without_rerank")
def search_without_rerank(req: SearchRequest):
    top_n = max(1, min(req.top_n, 20))  # защита: 1..20 

    results = search_vacancies_without_rerank(req.text)

    top_results = results[:top_n]

    scores = [r["score"] for r in top_results]
    percents = normalize_scores_to_percent(scores)

   
    response = []
    for r, p in zip(top_results[:5], percents):
        response.append({
            "vacancy_id": r["id"],
            "text": r["content"],
            "relevance_percent": p
        })

    return {
        "query": req.text,
        "results": response
    }    