import os
from typing import Optional
import numpy as np
from openai import OpenAI
from .base import EmbeddingProvider


class OpenAIEmbedding(EmbeddingProvider):
    def __init__(self, model_name: str = "text-embedding-3-small", api_key: Optional[str] = None):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        response = self.client.embeddings.create(model=self.model_name, input=texts)
        vectors = [item.embedding for item in response.data]
        return np.asarray(vectors, dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(model=self.model_name, input=[text])
        return np.asarray(response.data[0].embedding, dtype="float32")
