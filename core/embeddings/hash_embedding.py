import hashlib

import numpy as np

from .base import EmbeddingProvider


class HashEmbedding(EmbeddingProvider):
    """Deterministic local embedder for tests. Not intended as a quality baseline."""

    def __init__(self, dim: int = 32):
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.lower().encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little")
        rng = np.random.default_rng(seed)
        vector = rng.standard_normal(self.dim).astype("float32")
        norm = float(np.linalg.norm(vector)) or 1e-12
        return vector / norm

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        return np.vstack([self._vector(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)
