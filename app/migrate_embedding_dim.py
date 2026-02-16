"""
Миграция: изменение размерности embedding на 1024
(для модели intfloat/multilingual-e5-large).

Запуск: python -m app.migrate_embedding_dim
"""
from app.db import get_conn

conn = get_conn()
cur = conn.cursor()

# Очищаем старые эмбеддинги и меняем размерность колонки
# USING NULL — все старые значения станут NULL (они несовместимы с новой размерностью)
cur.execute("""
    ALTER TABLE messages
    ALTER COLUMN embedding TYPE vector(1024) USING NULL
""")

conn.commit()
cur.close()
conn.close()

print("Миграция выполнена: embedding изменён на vector(1024).")
print("Запустите embed_vacancies для пересчёта эмбеддингов.")
