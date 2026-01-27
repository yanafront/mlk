from typing import List
import math


def information_density(text: str) -> float:
    words = text.lower().split()
    if len(words) < 20:
        return 0.3

    unique_ratio = len(set(words)) / len(words)
    return min(1.0, unique_ratio * 2)


def length_score(text: str) -> float:
    l = len(text)
    if l < 80:
        return 0.2
    if l < 200:
        return 0.6
    return 1.0


def cosine_similarity(a, b):
    a = [float(x) for x in a]
    b = [float(x) for x in b]
    return sum(x * y for x, y in zip(a, b))



def embedding_confidence(
    vacancy_embedding: List[float],
    generic_embedding: List[float]
) -> float:
    sim = cosine_similarity(vacancy_embedding, generic_embedding)
    return max(0.0, 1.0 - sim)


def compute_confidence(
    text: str,
    vacancy_embedding,
    generic_embedding
) -> float:
    vacancy_embedding = [float(x) for x in vacancy_embedding]
    generic_embedding = [float(x) for x in generic_embedding]

    return (
        0.4 * information_density(text) +
        0.3 * length_score(text) +
        0.3 * embedding_confidence(vacancy_embedding, generic_embedding)
    )
