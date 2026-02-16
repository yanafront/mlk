# MLK — Job Semantic Search ML Service

Семантический поиск вакансий и подбор кандидатов на основе ML-моделей. Сервис использует векторные эмбеддинги (pgvector) и cross-encoder для ранжирования результатов.

## Возможности

- **Поиск вакансий** — семантический поиск по запросу кандидата с rerank-моделью
- **Подбор кандидатов** — поиск подходящих пользователей по тексту вакансии
- **Управление пользователями** — добавление и обновление профилей кандидатов с автоматическим созданием эмбеддингов
- **Эмбеддинги** — получение векторного представления текста через API

## Технологии

- **FastAPI** — веб-фреймворк
- **PostgreSQL + pgvector** — хранение векторов
- **sentence-transformers** — модели эмбеддингов (multilingual-e5-large)
- **CrossEncoder** — reranker (bge-reranker-v2-m3)
- **PyTorch** — CUDA для ускорения на GPU

## Требования

- Python 3.10+
- PostgreSQL с расширением pgvector
- NVIDIA GPU (рекомендуется) или CPU

## Установка

### 1. Клонирование и зависимости

```bash
pip install -r requirements.txt
```

### 2. Переменные окружения

Скопируйте `example.env` в `.env` и настройте:

```bash
cp example.env .env
```

| Переменная | Описание |
|------------|----------|
| `POSTGRES_HOST` | Хост PostgreSQL |
| `POSTGRES_PORT` | Порт (по умолчанию 5432) |
| `POSTGRES_DB` | Имя базы данных |
| `POSTGRES_USER` | Пользователь |
| `POSTGRES_PASSWORD` | Пароль |
| `TOP_K` | Максимум результатов в поиске (по умолчанию 50) |

### 3. Запуск

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Документация API: http://localhost:8000/docs

## Docker

```bash
docker-compose up --build
```

Сервис будет доступен на порту 8000. Для работы GPU требуется NVIDIA Container Toolkit.

## API Endpoints

| Метод | Endpoint | Описание |
|-------|----------|----------|
| POST | `/search` | Поиск вакансий по запросу (с rerank) |
| POST | `/search_without_rerank` | Поиск вакансий без rerank (быстрее) |
| POST | `/vacancy/match_users` | Подбор кандидатов по тексту вакансии |
| POST | `/users` | Добавление/обновление пользователя (кандидата) |
| POST | `/embed` | Получение эмбеддинга для текста |
| POST | `/query` | Сохранение запроса и поиск вакансий |

### Примеры запросов

**Поиск вакансий:**
```json
POST /search
{
  "text": "Python разработчик удалённо",
  "top_n": 5
}
```

**Подбор кандидатов по вакансии:**
```json
POST /vacancy/match_users
{
  "vacancy_text": "Ищем backend-разработчика на Python...",
  "top_n": 10
}
```

**Добавление пользователя:**
```json
POST /users
{
  "user_id": 12345,
  "description": "Опыт 3 года, Python, FastAPI, PostgreSQL..."
}
```

## Структура проекта

```
mlk/
├── app/
│   ├── main.py           # FastAPI приложение
│   ├── search.py         # Логика семантического поиска
│   ├── db.py             # Подключение к PostgreSQL
│   ├── models.py         # Загрузка ML-моделей
│   ├── schemas.py        # Pydantic-схемы
│   ├── settings.py       # Конфигурация моделей
│   ├── text_normalizer.py
│   ├── vacancy_normalizer.py
│   ├── confidence.py     # Расчёт confidence для вакансий
│   ├── embed_users.py    # Индексация пользователей
│   └── migrate_users.py  # Миграция таблицы пользователей
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── example.env
```

## Модели

- **Эмбеддинги:** `intfloat/multilingual-e5-large` — мультиязычная модель для query/passage
- **Reranker:** `BAAI/bge-reranker-v2-m3` — cross-encoder для переранжирования

Модели загружаются при первом запуске и кэшируются в `~/.cache/huggingface`.
