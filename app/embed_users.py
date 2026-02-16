"""
Индексация пользователей: создание embeddings для таблицы users.

E5-модель требует префикс "passage: " для документов (профилей кандидатов).

Запуск: python -m app.embed_users
С пересчётом всех: python -m app.embed_users --force
"""
import sys
from app.models import embedding_model
from app.db import get_conn
import psycopg2.extras

conn = get_conn()
cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

force_reembed = "--force" in sys.argv
where_clause = "" if force_reembed else "WHERE embedding IS NULL"

cur.execute(f"""
    SELECT id, content
    FROM users
    {where_clause}
""")

rows = cur.fetchall()

for row in rows:
    # E5 требует префикс "passage: " для документов
    text = (row["content"] or "").strip()
    if not text:
        continue

    emb = embedding_model.encode(
        f"passage: {text}",
        normalize_embeddings=True
    ).tolist()

    cur.execute(
        "UPDATE users SET embedding = %s WHERE id = %s",
        (emb, row["id"])
    )
    conn.commit()
    print(f"Processed user {row['id']}")

cur.close()
conn.close()
print(f"Готово. Обработано пользователей: {len(rows)}")
