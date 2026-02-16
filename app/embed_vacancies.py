"""
Индексация вакансий: создание embeddings для таблицы messages.

E5-модель требует префиксы: "query: " для запросов, "passage: " для документов.
После изменения префиксов нужно пересчитать embeddings: python -m app.embed_vacancies --force
"""
import sys
from app.models import embedding_model
from app.db import get_conn
from app.text_normalizer import normalize_vacancy
from app.vacancy_normalizer import normalize_vacancy_llm, normalized_data_to_embedding_text
import psycopg2.extras

conn = get_conn()
cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

force_reembed = "--force" in sys.argv
where_clause = "" if force_reembed else "WHERE embedding IS NULL"

cur.execute(f"""
    SELECT id, content, normalized
    FROM messages
    {where_clause}
""")

rows = cur.fetchall()

for row in rows:
    normalized_data = row.get("normalized") or normalize_vacancy_llm(row["content"])
    text = normalized_data_to_embedding_text(normalized_data) or normalize_vacancy(row["content"])
    if isinstance(normalized_data, dict) and "error" not in normalized_data and not row.get("normalized"):
        cur.execute(
            "UPDATE messages SET normalized = %s WHERE id = %s",
            (psycopg2.extras.Json(normalized_data), row["id"])
        )
    # E5 требует префикс "passage: " для документов (иначе качество сильно падает)
    emb = embedding_model.encode(
        f"passage: {text}",
        normalize_embeddings=True
    ).tolist()

    cur.execute(
        "UPDATE messages SET embedding = %s WHERE id = %s",
        (emb, row["id"])
    )

    conn.commit()
    print(f"Processed {row['id']}")
    
cur.close()
conn.close()
