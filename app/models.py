import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from app.settings import DEVICE, EMBEDDING_MODEL, RERANKER_MODEL

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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

