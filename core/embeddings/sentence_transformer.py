import numpy as np

from .base import EmbeddingProvider


class SentenceTransformerEmbedding(EmbeddingProvider):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("Sentence Transformers is not installed. Choose another embedding provider or install sentence-transformers.") from exc
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0], dtype="float32")
