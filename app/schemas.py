from pydantic import BaseModel
from typing import List

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
