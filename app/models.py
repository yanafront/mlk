from sentence_transformers import SentenceTransformer, CrossEncoder
from app.settings import DEVICE, EMBEDDING_MODEL, RERANKER_MODEL

print("Loading embedding model...")
embedding_model = SentenceTransformer(
    EMBEDDING_MODEL,
    device=DEVICE
)

print("Loading reranker model...")
reranker_model = CrossEncoder(
    RERANKER_MODEL,
    device=DEVICE
)

print("Models loaded.")

GENERIC_VACANCY_TEXT = (
    "Описание вакансии без указания обязанностей, "
    "требований, профессии и условий работы."
)

generic_vacancy_embedding = embedding_model.encode(
    GENERIC_VACANCY_TEXT,
    normalize_embeddings=True
).tolist()

