"""
Миграция: создание таблицы users для хранения профилей кандидатов с embeddings.

Запуск: python -m app.migrate_users

Требует расширение pgvector в БД.
"""
from app.db import get_conn

conn = get_conn()
cur = conn.cursor()

cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        content TEXT NOT NULL,
        embedding vector(1024)
    )
""")

conn.commit()
cur.close()
conn.close()

print("Миграция выполнена: таблица users создана.")
