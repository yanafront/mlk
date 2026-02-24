from pydantic import BaseModel
from typing import List, Optional

class EmbedRequest(BaseModel):
    text: str

class EmbedResponse(BaseModel):
    embedding: List[float]

class RerankRequest(BaseModel):
    query: str
    documents: List[str]

class RerankResult(BaseModel):
    document: str
    score: float

class RerankResponse(BaseModel):
    results: List[RerankResult]


class VacancyMatchRequest(BaseModel):
    """Запрос на поиск пользователей по вакансии."""
    vacancy_text: str
    top_n: int = 10


class AddUserRequest(BaseModel):
    """Добавление пользователя (профиль/резюме)."""
    description: str
    user_id: int


class SearchFilters(BaseModel):
    """SQL-фильтры по normalized полям для поиска вакансий."""
    location: Optional[str] = None  # Фильтр по городу/местоположению
    employment_type: Optional[str] = None  # Тип занятости (полный день, удалённо и т.д.)
    occupation: Optional[str] = None  # Категория профессии (IT, Продажи и т.д.)


class SearchRequest(BaseModel):
    text: str
    top_n: int = 5
    filters: Optional[SearchFilters] = None  # Опциональные SQL-фильтры
