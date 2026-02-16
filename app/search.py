from typing import List, Dict, Any

import os
import time
import psycopg2.extras
import json

from app.models import embedding_model, reranker_model
from app.db import get_conn
from app.text_normalizer import normalize_vacancy
from app.vacancy_normalizer import normalize_vacancy_llm, normalized_data_to_embedding_text
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

    # E5 требует префикс "query: " для запросов (иначе качество сильно падает)
    query_embedding = embedding_model.encode(
        f"query: {user_query}",
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
            normalized,
            embedding,
            embedding <=> %s::vector AS distance
        FROM messages
        WHERE embedding IS NOT NULL
        ORDER BY distance
        LIMIT 100;
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
        normalized_data_to_embedding_text(r["normalized"]) or normalize_vacancy(r["content"])
        for r in rows
    ]

    pairs = [
        (f"query: {user_query}", f"passage: {doc}")
        for doc in documents
    ]

    rerank_scores = reranker_model.predict(pairs, batch_size=16)
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

        if final_score >= 0.3:
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
    

def search_vacancies_without_rerank(user_query: str) -> List[Dict[str, Any]]:
    t_start = time.perf_counter()
    metrics = {}

    # ---------- 1. EMBEDDING ЗАПРОСА ----------
    t0 = time.perf_counter()
    # E5 требует префикс "query: " для запросов (иначе качество сильно падает)
    query_embedding = embedding_model.encode(
        f"query: {user_query}",
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
            normalized,
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
        print("search_vacancies_without_rerank metrics:", metrics)
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
        print("search_vacancies_without_rerank metrics:", metrics)
        return []

    # ---------- 4. FINAL SCORE = (1 - distance) × confidence ----------
    t0 = time.perf_counter()
    results = []

    for row in rows:
        semantic_score = max(0.0, 1.0 - float(row["distance"]))
        vacancy_embedding = parse_pgvector(row["embedding"])

        confidence = compute_confidence(
            text=row["content"],
            vacancy_embedding=vacancy_embedding,
            generic_embedding=generic_vacancy_embedding
        )

        final_score = float(semantic_score) * confidence

        if final_score >= 0.5:
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
    print("search_vacancies_without_rerank metrics:", metrics)

    return results


def search_users_by_vacancy(vacancy_text: str, top_k: int = 20) -> List[Dict[str, Any]]:
    """
    По вакансии находит подходящих пользователей (кандидатов).
    Вакансия — запрос, профили пользователей — документы.
    Вакансия нормализуется так же, как в embed_vacancies.
    """
    t_start = time.perf_counter()
    metrics = {}

    # ---------- 0. НОРМАЛИЗАЦИЯ ВАКАНСИИ (как в embed_vacancies) ----------
    t0 = time.perf_counter()
    normalized_data = normalize_vacancy_llm(vacancy_text)
    query_text = normalized_data_to_embedding_text(normalized_data) or normalize_vacancy(vacancy_text)
    metrics["normalize_ms"] = (time.perf_counter() - t0) * 1000

    # ---------- 1. EMBEDDING ВАКАНСИИ (как запрос) ----------
    t0 = time.perf_counter()
    vacancy_embedding = embedding_model.encode(
        f"query: {query_text}",
        normalize_embeddings=True
    ).tolist()
    metrics["embedding_ms"] = (time.perf_counter() - t0) * 1000

    # ---------- 2. VECTOR SEARCH В POSTGRES (таблица users) ----------
    t0 = time.perf_counter()
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cur.execute(
        """
        SELECT
            id,
            description,
            embedding,
            embedding <=> %s::vector AS distance
        FROM main
        WHERE embedding IS NOT NULL
        ORDER BY distance
        LIMIT 100;
        """,
        (vacancy_embedding,)
    )

    rows = cur.fetchall()
    cur.close()
    conn.close()
    metrics["vector_search_ms"] = (time.perf_counter() - t0) * 1000

    if not rows:
        metrics["total_ms"] = (time.perf_counter() - t_start) * 1000
        print("search_users_by_vacancy metrics:", metrics)
        return []

    # ---------- 3. RERANK: вакансия vs профили пользователей ----------
    t0 = time.perf_counter()
    pairs = [
        (f"query: {query_text}", f"passage: {r['description']}")
        for r in rows
    ]
    rerank_scores = reranker_model.predict(pairs, batch_size=16)
    metrics["rerank_ms"] = (time.perf_counter() - t0) * 1000

    # ---------- 4. ФОРМИРУЕМ РЕЗУЛЬТАТЫ ----------
    results = []
    for row, score in zip(rows, rerank_scores):
        if float(score) >= 0.3:
            results.append({
                "id": row["id"],
                "description": row["description"],
                "score": float(score)
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    results = results[:top_k]

    metrics["total_ms"] = (time.perf_counter() - t_start) * 1000
    metrics["candidates_count"] = len(rows)
    metrics["results_count"] = len(results)
    print("search_users_by_vacancy metrics:", metrics)

    return results