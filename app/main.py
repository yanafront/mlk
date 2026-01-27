from fastapi import FastAPI
from pydantic import BaseModel
import math
from app.search import search_vacancies
from app.db import get_conn
from app.models import embedding_model, reranker_model
from app.schemas import (
    EmbedRequest,
)

app = FastAPI(title="Job Semantic Search ML Service")


class SearchRequest(BaseModel):
    text: str
    top_n: int = 5 


def score_to_percent(score: float) -> int:
    return int(100 / (1 + math.exp(-score)))

def normalize_scores_to_percent(scores):
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
    text = f"Поиск вакансий: {req.text}"
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