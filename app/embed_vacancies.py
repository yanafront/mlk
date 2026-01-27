from app.models import embedding_model
from app.db import get_conn
from app.text_normalizer import normalize_vacancy
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
    text = normalize_vacancy(row["content"])
    emb = embedding_model.encode(
        text,
        normalize_embeddings=True
    ).tolist()

    cur.execute(
        "UPDATE messages SET embedding = %s WHERE id = %s",
        (emb, row["id"])
    )

conn.commit()
cur.close()
conn.close()
