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
