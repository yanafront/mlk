from typing import List, Dict, Any

import os
import time
import psycopg2.extras
import json

from app.models import embedding_model, reranker_model
from app.db import get_conn
from app.text_normalizer import normalize_vacancy
from app.vacancy_normalizer import normalize_vacancy_llm, normalized_data_to_embedding_text
from app.confidence import compute_confidence, get_generic_vacancy_embedding


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


def search_vacancies(
    user_query: str,
) -> List[Dict[str, Any]]:
    import os
    VECTOR_K = int(os.getenv("VECTOR_K", 200))
    RERANK_K = int(os.getenv("RERANK_K", 20))
    FINAL_K = int(os.getenv("FINAL_K", 5))

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

    where_conditions = ["embedding IS NOT NULL"]
    query_params = [query_embedding]

    where_clause = " AND ".join(where_conditions)
    query_params.append(VECTOR_K)

    cur.execute(f"""
    SELECT
        id,
        content,
        normalized,
        embedding,
        embedding <=> %s::vector AS distance
    FROM messages
    WHERE {where_clause}
    ORDER BY distance
    LIMIT %s;
    """, tuple(query_params))

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
    # Pre-rerank pruning: ограничиваем количество кандидатов для CrossEncoder
    rows = rows[:RERANK_K]
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
        (user_query, doc)
        for doc in documents
    ]

    rerank_scores = reranker_model.predict(pairs, batch_size=16)
    metrics["rerank_ms"] = (time.perf_counter() - t0) * 1000

    # ---------- 5. FINAL SCORE = semantic × confidence ----------
    t0 = time.perf_counter()
    results = []
    
    # Порог фильтрации. По умолчанию 0 = возвращаем топ по score без отсечения
    SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0"))
    
    # Отладочная информация для диагностики
    debug_scores = []

    generic_embedding = get_generic_vacancy_embedding()

    for row, semantic_score in zip(rows, rerank_scores):
        vacancy_embedding = parse_pgvector(row["embedding"])
        confidence = compute_confidence(
            text=row["content"],
            vacancy_embedding=vacancy_embedding,
            generic_embedding=generic_embedding
        )
        final_score = float(semantic_score) * (0.7 + 0.3 * confidence)

        # Сохраняем для отладки
        debug_scores.append({
            "id": row["id"],
            "semantic_score": float(semantic_score),
            "confidence": confidence,
            "final_score": final_score
        })

        if final_score >= SCORE_THRESHOLD:
            results.append({
                "id": row["id"],
                "content": row["content"],
                "score": final_score
            })
    
    RETURN_TOP_WHEN_BELOW = os.getenv("RETURN_TOP_WHEN_ALL_BELOW_THRESHOLD", "false").lower() == "true"
    if not results and debug_scores and RETURN_TOP_WHEN_BELOW:
        print(f"Все кандидаты ниже порога {SCORE_THRESHOLD}; возвращаем топ-{FINAL_K} по score (реальные документы из БД)")
        top_by_score = sorted(debug_scores, key=lambda x: x["final_score"], reverse=True)[:FINAL_K]
        for s in top_by_score:
            row = next(r for r in rows if r["id"] == s["id"])
            results.append({
                "id": row["id"],
                "content": row["content"],
                "score": s["final_score"]
            })

    # Диагностика: полная таблица scores (запуск с DIAGNOSE=1)
    if os.getenv("DIAGNOSE") == "1" and debug_scores:
        sorted_debug = sorted(debug_scores, key=lambda x: x["final_score"], reverse=True)
        print("DIAGNOSE scores (semantic × confidence = final):")
        for i, s in enumerate(sorted_debug[:20], 1):
            print(f"  {i}. id={s['id']} semantic={s['semantic_score']:.4f} conf={s['confidence']:.4f} final={s['final_score']:.4f}")
        if len(debug_scores) > 20:
            print(f"  ... и ещё {len(debug_scores) - 20} кандидатов")
    
    metrics["confidence_ms"] = (time.perf_counter() - t0) * 1000

    # ---------- 6. SORT И ФИНАЛЬНЫЙ ФИЛЬТР ----------
    t0 = time.perf_counter()
    results.sort(key=lambda x: x["score"], reverse=True)
    # FINAL_K — конечное количество результатов после rerank + confidence
    results = results[:FINAL_K]
    metrics["sort_ms"] = (time.perf_counter() - t0) * 1000

    metrics["total_ms"] = (time.perf_counter() - t_start) * 1000
    metrics["candidates_count"] = len(rows)
    metrics["results_count"] = len(results)
    print(
        "search_vacancies sizes:",
        "vector_candidates=", len(rows),
        "final_results=", len(results),
    )
    print("search_vacancies metrics:", metrics)

    return results
    

def search_vacancies_without_rerank(
    user_query: str,
) -> List[Dict[str, Any]]:
    import os
    VECTOR_K = int(os.getenv("VECTOR_K", 200))
    RERANK_K = int(os.getenv("RERANK_K", 20))
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

    where_conditions = ["embedding IS NOT NULL"]
    query_params = [query_embedding]

    where_clause = " AND ".join(where_conditions)
    query_params.append(VECTOR_K)

    cur.execute(
        f"""
        SELECT
            id,
            content,
            normalized,
            embedding,
            embedding <=> %s::vector AS distance
        FROM messages
        WHERE {where_clause}
        ORDER BY distance
        LIMIT %s;
        """,
        tuple(query_params)
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
    rows = rows[:RERANK_K]
    
    metrics["filter_ms"] = (time.perf_counter() - t0) * 1000

    if not rows:
        metrics["total_ms"] = (time.perf_counter() - t_start) * 1000
        print("search_vacancies_without_rerank metrics:", metrics)
        return []

    # ---------- 4. FINAL SCORE = (1 - distance) × confidence ----------
    t0 = time.perf_counter()
    results = []
    generic_embedding = get_generic_vacancy_embedding()
    for row in rows:
        semantic_score = max(0.0, 1.0 - float(row["distance"]))
        vacancy_embedding = parse_pgvector(row["embedding"])

        confidence = compute_confidence(
            text=row["content"],
            vacancy_embedding=vacancy_embedding,
            generic_embedding= generic_embedding
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
    import os
    VECTOR_K = int(os.getenv("VECTOR_K", 200))
    RERANK_K = int(os.getenv("RERANK_K", 20))
    FINAL_K = int(os.getenv("FINAL_K", 5))
    USE_LLM_NORMALIZE = os.getenv("USE_LLM_NORMALIZE", "true").lower() == "true"

    t_start = time.perf_counter()
    metrics = {}

        # ---------- 0. НОРМАЛИЗАЦИЯ ВАКАНСИИ ----------
    t0 = time.perf_counter()
    USE_LLM_NORMALIZE = os.getenv("USE_LLM_NORMALIZE", "true").lower() == "true"
    
    if USE_LLM_NORMALIZE:
        normalized_data = normalize_vacancy_llm(vacancy_text)
        query_text = normalized_data_to_embedding_text(normalized_data) or normalize_vacancy(vacancy_text)
    else:
        query_text = normalize_vacancy(vacancy_text)
        normalized_data = None
    metrics["normalize_ms"] = (time.perf_counter() - t0) * 1000

    # ---------- 1. EMBEDDING ВАКАНСИИ ----------
    t0 = time.perf_counter()
    vacancy_embedding = embedding_model.encode(
        f"query: {query_text}",
        normalize_embeddings=True
    ).tolist()
    passage_embedding = embedding_model.encode(
        f"passage: {query_text}",
        normalize_embeddings=True
    ).tolist()
    metrics["embedding_ms"] = (time.perf_counter() - t0) * 1000

    # ---------- 2. СОХРАНЕНИЕ ВАКАНСИИ В БАЗУ ----------
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cur.execute(
        """
        INSERT INTO messages (content, normalized, embedding)
        VALUES (%s, %s, %s)
        RETURNING id;
        """,
        (
            vacancy_text,
            psycopg2.extras.Json(normalized_data) if isinstance(normalized_data, dict) and "error" not in normalized_data else None,
            passage_embedding,
        ),
    )
    saved_vacancy_id = cur.fetchone()["id"]
    conn.commit()
    metrics["vacancy_id"] = saved_vacancy_id

    # ---------- 3. VECTOR SEARCH В POSTGRES ----------
    t0 = time.perf_counter()
    cur.execute(
        """
        SELECT
            user_id,
            description,
            embedding,
            embedding <=> %s::vector AS distance
        FROM users
        WHERE embedding IS NOT NULL
        ORDER BY distance
        LIMIT %s;
        """,
        (vacancy_embedding, VECTOR_K)
    )

    rows = cur.fetchall()
    cur.close()
    conn.close()
    metrics["vector_search_ms"] = (time.perf_counter() - t0) * 1000

    if not rows:
        metrics["total_ms"] = (time.perf_counter() - t_start) * 1000
        print("search_users_by_vacancy metrics:", metrics)
        return []

    # ---------- 4. PRE-RERANK PRUNING ----------
    rows = rows[:RERANK_K]

    # ---------- 5. RERANK: вакансия vs профили пользователей ----------
    t0 = time.perf_counter()
    pairs = [
        (query_text, r['description'])
        for r in rows
    ]
    rerank_scores = reranker_model.predict(pairs, batch_size=16)
    metrics["rerank_ms"] = (time.perf_counter() - t0) * 1000

    # ---------- 6. ФОРМИРУЕМ РЕЗУЛЬТАТЫ ----------
    results = []
    for row, score in zip(rows, rerank_scores):
        print("score:", score)
        if float(score) >= 0.01:
            results.append({
                "user_id": row["user_id"],
                "description": row["description"],
                "score": float(score)
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    results = results[: min(top_k, FINAL_K)]

    metrics["total_ms"] = (time.perf_counter() - t_start) * 1000
    metrics["candidates_count"] = len(rows)
    metrics["results_count"] = len(results)
    print(
        "search_users_by_vacancy sizes:",
        "vector_candidates=", len(rows),
        "final_results=", len(results),
    )
    print("search_users_by_vacancy metrics:", metrics)
    
    return results