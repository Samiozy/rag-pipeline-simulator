import os
from typing import Optional

import numpy as np

from .base import EmbeddingProvider


class OpenAIEmbedding(EmbeddingProvider):
    def __init__(self, model_name: str = "text-embedding-3-small", api_key: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("The OpenAI package is not installed.") from exc
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("An OpenAI API key is required for OpenAI embeddings.")
        self.model_name = model_name
        self.client = OpenAI(api_key=key)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        response = self.client.embeddings.create(model=self.model_name, input=texts)
        vectors = [item.embedding for item in response.data]
        return np.asarray(vectors, dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(model=self.model_name, input=[text])
        return np.asarray(response.data[0].embedding, dtype="float32")
