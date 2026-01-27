from typing import List, Dict, Any

import os
import psycopg2.extras
import json

from app.models import embedding_model, reranker_model
from app.db import get_conn
from app.text_normalizer import normalize_vacancy
from app.confidence import compute_confidence
from app.models import generic_vacancy_embedding


TOP_K = int(os.getenv("TOP_K", 50))
FINAL_K = 10


def parse_pgvector(raw_embedding) -> List[float]:
    """
    Приводит embedding из pgvector к List[float]
    psycopg2 может вернуть:
    - строку "[0.1, 0.2, ...]"
    - список Decimal
    """
    if raw_embedding is None:
        return []

    if isinstance(raw_embedding, str):
        return [float(x) for x in json.loads(raw_embedding)]

    # list / tuple / Decimal[]
    return [float(x) for x in raw_embedding]

def is_valid_vacancy(text: str) -> bool:
    """
    Фильтр мусорных вакансий:
    слишком короткие / пустые
    """
    if not text:
        return False
    return len(text.strip()) >= 50


def search_vacancies(user_query: str) -> List[Dict[str, Any]]:
    # ---------- 1. EMBEDDING ЗАПРОСА ----------
    query_text = (
        "Задача: найти подходящую вакансию по запросу кандидата.\n"
        f"Запрос пользователя: {user_query}"
    )

    query_embedding = embedding_model.encode(
        query_text,
        normalize_embeddings=True
    ).tolist()

    # ---------- 2. VECTOR SEARCH В POSTGRES ----------
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cur.execute(
        """
        SELECT
            id,
            content,
            embedding,
            embedding <=> %s::vector AS distance
        FROM messages
        WHERE embedding IS NOT NULL
        ORDER BY distance
        LIMIT %s
        """,
        (query_embedding, TOP_K)
    )

    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        return []

    # ---------- 3. ФИЛЬТР МУСОРА ----------
    rows = [
        r for r in rows
        if is_valid_vacancy(r["content"])
    ]

    if not rows:
        return []

    # ---------- 4. RERANK ----------
    documents = [
        normalize_vacancy(r["content"])
        for r in rows
    ]

    pairs = [
        (
            f"Запрос пользователя: {user_query}",
            doc
        )
        for doc in documents
    ]

    rerank_scores = reranker_model.predict(pairs)

    # ---------- 5. FINAL SCORE = semantic × confidence ----------
    results = []

    for row, semantic_score in zip(rows, rerank_scores):
        vacancy_embedding = parse_pgvector(row["embedding"])

        confidence = compute_confidence(
            text=row["content"],
            vacancy_embedding=vacancy_embedding,
            generic_embedding=generic_vacancy_embedding
        )

        final_score = float(semantic_score) * confidence

        results.append({
            "id": row["id"],
            "content": row["content"],
            "score": final_score
        })

    # ---------- 6. SORT ----------
    results.sort(key=lambda x: x["score"], reverse=True)

    return results