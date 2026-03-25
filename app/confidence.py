from typing import List
import math
from app.models import embedding_model

GENERIC_VACANCY_TEXT = (
    "Описание вакансии без указания обязанностей, "
    "требований, профессии и условий работы."
)

_generic_vacancy_embedding = None

def get_generic_vacancy_embedding() -> List[float]:
    """Возвращает единожды сгенерированный generic embedding"""
    global _generic_vacancy_embedding
    if _generic_vacancy_embedding is None:
        _generic_vacancy_embedding = embedding_model.encode(
            GENERIC_VACANCY_TEXT,
            normalize_embeddings=True
        ).tolist()
    return _generic_vacancy_embedding

def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x*y for x,y in zip(a,b))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(x*x for x in b))
    if norm_a==0 or norm_b==0:
        return 0.0
    return dot / (norm_a * norm_b)

def embedding_confidence(vacancy_embedding: List[float], generic_embedding: List[float]) -> float:
    sim = cosine_similarity(vacancy_embedding, generic_embedding)
    return max(0.0, min(1.0, 1.0 - sim))

def compute_confidence(text: str, vacancy_embedding: List[float], generic_embedding: List[float]) -> float:
    if not text.strip() or not vacancy_embedding:
        return 0.0
    words = [w.lower() for w in text.split() if w.isalpha()]
    if len(words) < 5:
        return 0.2
    info_score = 0.4 + 0.6 * len(set(words)) / len(words)
    len_score = 0.4 if len(text.strip())<80 else 0.7 if len(text.strip())<200 else 1.0
    embed_score = embedding_confidence(vacancy_embedding, generic_embedding)
    confidence = 0.5*info_score + 0.3*len_score + 0.2*embed_score
    confidence = max(confidence, 0.3)
    return min(max(confidence,0.0),1.0)