from typing import List, Dict, Any

import os
import time
import psycopg2.extras
import json

from app.models import embedding_model, reranker_model
from app.db import get_conn
from app.vacancy_normalizer import (
    normalize_vacancy_llm,
    normalized_data_to_embedding_text,
    has_usable_normalized_vacancy_data,
)
from app.confidence import compute_confidence, get_generic_vacancy_embedding
from app.text_normalizer import normalize_vacancy
from app.vacancy_normalizer import normalized_data_to_embedding_text

TOP_K = int(os.getenv("TOP_K", 50))
VECTOR_K = int(os.getenv("VECTOR_K", 200))
RERANK_K = int(os.getenv("RERANK_K", 20))
FINAL_K = int(os.getenv("FINAL_K", 5))  
USE_LLM_NORMALIZE = os.getenv("USE_LLM_NORMALIZE", "true").lower() == "true"

def extract_filters(query: str):
    """Извлекает профессию и город из запроса"""
    q = query.lower()
    
    professions = ["официант", "курьер", "продавец", "программист", 
                   "бариста", "повар", "администратор", "водитель"]
    cities = ["минск", "гомель", "могилёв", "витебск", "гродно", "брест"]
    
    found_prof = next((p for p in professions if p in q), None)
    found_city = next((c for c in cities if c in q), None)
    
    return found_prof, found_city

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
    import os
    import time
    import psycopg2.extras
    
    VECTOR_K = int(os.getenv("VECTOR_K", 300))
    RERANK_K = int(os.getenv("RERANK_K", 15))
    FINAL_K = int(os.getenv("FINAL_K", 10))
    
    t_start = time.perf_counter()
    
    # ---------- 1. QUERY ----------
    query_embedding = embedding_model.encode(
        f"query: {user_query}",
        normalize_embeddings=True
    ).tolist()
    
    # ---------- 2. VECTOR SEARCH ----------
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    
    cur.execute("""
        SELECT id, content, normalized, embedding,
               embedding <=> %s::vector AS distance
        FROM messages
        WHERE embedding IS NOT NULL
        ORDER BY distance
        LIMIT %s;
    """, (query_embedding, VECTOR_K))
    
    rows = cur.fetchall()
    cur.close()
    conn.close()
    
    if not rows:
        return []
    
    # ---------- 3. МЯГКИЕ ФИЛЬТРЫ ----------
    profession, city = extract_filters(user_query)
    
    for r in rows:
        text_lower = r["content"].lower()
        
        r["profession_bonus"] = 1.2 if (profession and profession in text_lower) else 1.0
        
        if city and city in text_lower:
            r["city_bonus"] = 1.2
        elif city:
            r["city_bonus"] = 0.9
        else:
            r["city_bonus"] = 1.0
    
    rows = rows[:RERANK_K]
    
    # ---------- 4. ДОКУМЕНТЫ ----------
    documents = []
    for r in rows:
        base_text = normalize_vacancy(r["content"])
        norm_text = normalized_data_to_embedding_text(r["normalized"])
        
        doc = norm_text if norm_text and len(norm_text) >= 30 else base_text
        documents.append(doc[:512])
    
    # ---------- 5. RERANK ----------
    pairs = [(f"query: {user_query}", f"passage: {doc}") for doc in documents]
    rerank_scores = reranker_model.predict(pairs, batch_size=16)
    
    # ---------- 6. SCORE ----------
    results = []
    for r, score in zip(rows, rerank_scores):
        final_score = float(score)
        final_score *= r["profession_bonus"]
        final_score *= r["city_bonus"]
        
        if user_query.lower() in r["content"].lower():
            final_score *= 1.2
        
        results.append({
            "id": r["id"],
            "content": r["content"],
            "score": final_score
        })
    
    # ---------- 7. СОРТИРОВКА ----------
    results.sort(key=lambda x: x["score"], reverse=True)
    results = results[:FINAL_K]
    
    print(f"Search completed in {round((time.perf_counter() - t_start)*1000, 2)} ms")
    
    return results
    

def search_vacancies_without_rerank(
    user_query: str,
) -> List[Dict[str, Any]]:

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

    # ---------- 4. FINAL SCORE ----------
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


def search_users_by_vacancy(vacancy_text: str, top_k: int = 20) -> Dict[str, Any]:
    """
    По вакансии находит подходящих пользователей (кандидатов).
    Вакансия — запрос, профили пользователей — документы.
    Вакансия нормализуется так же, как в embed_vacancies.
    """

    t_start = time.perf_counter()
    metrics = {}

    # ---------- 0. НОРМАЛИЗАЦИЯ ВАКАНСИИ ----------
    t0 = time.perf_counter()

    if USE_LLM_NORMALIZE:
        normalized_data = normalize_vacancy_llm(vacancy_text)
    else:
        normalized_data = None
    query_text = normalized_data_to_embedding_text(normalized_data)
    metrics["normalize_ms"] = (time.perf_counter() - t0) * 1000

    if USE_LLM_NORMALIZE and not has_usable_normalized_vacancy_data(normalized_data):
        metrics["total_ms"] = (time.perf_counter() - t_start) * 1000
        print("search_users_by_vacancy: skip (no normalized data), metrics:", metrics)
        return {
            "query_text": normalized_data,
            "results": [],
        }

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
        FROM main
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
        return {
            "query_text": normalized_data,
            "results": [],
        }

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
        if float(score) >= 0.065:
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

    return {
        "query_text": normalized_data,
        "results": results,
    }