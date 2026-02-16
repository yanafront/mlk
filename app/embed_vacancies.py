from app.models import embedding_model
from app.db import get_conn
from app.text_normalizer import normalize_vacancy
from app.vacancy_normalizer import normalize_vacancy_llm, normalized_data_to_embedding_text
import psycopg2.extras

conn = get_conn()
cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

cur.execute("""
    SELECT id, content
    FROM messages
    WHERE embedding IS NULL
""")

rows = cur.fetchall()

for row in rows:
    normalized_data = normalize_vacancy_llm(row["content"])
    text = normalized_data_to_embedding_text(normalized_data) or normalize_vacancy(row["content"])
    cur.execute(
        "UPDATE messages SET normalized = %s WHERE id = %s",
        (psycopg2.extras.Json(normalized_data), row["id"])
    )
    emb = embedding_model.encode(
        text,
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
