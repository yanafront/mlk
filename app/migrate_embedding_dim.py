"""
Миграция: изменение размерности embedding с 1024 на 384
(для модели paraphrase-multilingual-MiniLM-L12-v2).

Запуск: python -m app.migrate_embedding_dim
"""
from app.db import get_conn

conn = get_conn()
cur = conn.cursor()

# Очищаем старые эмбеддинги и меняем размерность колонки
# USING NULL — все старые значения станут NULL (они несовместимы с новой размерностью)
cur.execute("""
    ALTER TABLE messages
    ALTER COLUMN embedding TYPE vector(384) USING NULL
""")

conn.commit()
cur.close()
conn.close()

print("Миграция выполнена: embedding изменён на vector(384).")
print("Запустите embed_vacancies для пересчёта эмбеддингов.")
