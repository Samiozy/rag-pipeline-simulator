import faiss
import numpy as np
from models import Chunk, RetrievalResult
from .base import VectorStore


class FAISSVectorStore(VectorStore):
    def __init__(self):
        self.index = None
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match")
        if not chunks:
            self.clear()
            return
        vectors = np.asarray(embeddings, dtype="float32")
        faiss.normalize_L2(vectors)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.chunks = list(chunks)

    def search(self, query_embedding: np.ndarray, k: int = 5, threshold: float = 0.0) -> list[RetrievalResult]:
        if self.index is None or not self.chunks:
            return []
        query = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(query)
        k = min(k, len(self.chunks))
        scores, indices = self.index.search(query, k)
        results: list[RetrievalResult] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or float(score) < threshold:
                continue
            results.append(RetrievalResult(chunk=self.chunks[int(idx)], score=float(score), rank=len(results) + 1))
        return results

    def clear(self) -> None:
        self.index = None
        self.chunks = []
