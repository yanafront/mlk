from typing import List, Dict, Any

import os
import time
import psycopg2.extras
import json

from app.models import embedding_model, reranker_model
from app.db import get_conn
from app.text_normalizer import normalize_vacancy
from app.confidence import compute_confidence
from app.models import generic_vacancy_embedding


# Ограничение применяется после финального расчёта (rerank + confidence)
TOP_K = int(os.getenv("TOP_K", 50))


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
    t_start = time.perf_counter()
    metrics = {}

    # ---------- 1. EMBEDDING ЗАПРОСА ----------
    t0 = time.perf_counter()
    query_text = (
        "Задача: найти подходящую вакансию по запросу кандидата.\n"
        f"Запрос пользователя: {user_query}"
    )

    query_embedding = embedding_model.encode(
        user_query,
        normalize_embeddings=True
    ).tolist()
    metrics["embedding_ms"] = (time.perf_counter() - t0) * 1000

    # ---------- 2. VECTOR SEARCH В POSTGRES ----------
    t0 = time.perf_counter()
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
        LIMIT 1000;
        """,
        (query_embedding,)
    )

    rows = cur.fetchall()
    cur.close()
    conn.close()
    metrics["vector_search_ms"] = (time.perf_counter() - t0) * 1000

    if not rows:
        metrics["total_ms"] = (time.perf_counter() - t_start) * 1000
        print("search_vacancies metrics:", metrics)
        return []

    # ---------- 3. ФИЛЬТР МУСОРА ----------
    t0 = time.perf_counter()
    rows = [
        r for r in rows
        if is_valid_vacancy(r["content"])
    ]
    metrics["filter_ms"] = (time.perf_counter() - t0) * 1000

    if not rows:
        metrics["total_ms"] = (time.perf_counter() - t_start) * 1000
        print("search_vacancies metrics:", metrics)
        return []

    # ---------- 4. RERANK ----------
    t0 = time.perf_counter()
    documents = [
        normalize_vacancy(r["content"])
        for r in rows
    ]

    pairs = [
        (
        f"query: {user_query}",
        f"passage: {doc}"
    )
        for doc in documents
    ]

    rerank_scores = reranker_model.predict(pairs)
    metrics["rerank_ms"] = (time.perf_counter() - t0) * 1000

    # ---------- 5. FINAL SCORE = semantic × confidence ----------
    t0 = time.perf_counter()
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
    metrics["confidence_ms"] = (time.perf_counter() - t0) * 1000

    # ---------- 6. SORT И ФИНАЛЬНЫЙ ФИЛЬТР ----------
    t0 = time.perf_counter()
    results.sort(key=lambda x: x["score"], reverse=True)
    results = results[:TOP_K]
    metrics["sort_ms"] = (time.perf_counter() - t0) * 1000

    metrics["total_ms"] = (time.perf_counter() - t_start) * 1000
    metrics["candidates_count"] = len(rows)
    metrics["results_count"] = len(results)
    print("search_vacancies metrics:", metrics)

    return results