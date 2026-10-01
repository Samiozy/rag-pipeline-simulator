from typing import Optional

import numpy as np

from core.models import Chunk, RetrievalResult
from .base import VectorStore


class NumpyVectorStore(VectorStore):
    """Portable cosine-similarity backend with no dedicated vector DB dependency."""

    def __init__(self):
        self.embeddings: Optional[np.ndarray] = None
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match")
        vectors = np.asarray(embeddings, dtype="float32")
        if vectors.size == 0:
            self.clear()
            return
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self.embeddings = vectors / np.clip(norms, 1e-12, None)
        self.chunks = list(chunks)

    def search(self, query_embedding: np.ndarray, k: int = 5, threshold: float = 0.0) -> list[RetrievalResult]:
        if self.embeddings is None or not self.chunks:
            return []
        query = np.asarray(query_embedding, dtype="float32")
        query = query / max(float(np.linalg.norm(query)), 1e-12)
        scores = self.embeddings @ query
        indices = np.argsort(scores)[::-1][:k]
        results: list[RetrievalResult] = []
        for idx in indices:
            score = float(scores[idx])
            if score < threshold:
                continue
            results.append(RetrievalResult(self.chunks[int(idx)], score, len(results) + 1, channel="dense"))
        return results

    def clear(self) -> None:
        self.embeddings = None
        self.chunks = []
