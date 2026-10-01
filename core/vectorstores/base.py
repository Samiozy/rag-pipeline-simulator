from abc import ABC, abstractmethod

import numpy as np

from core.models import Chunk, RetrievalResult


class VectorStore(ABC):
    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int = 5, threshold: float = 0.0) -> list[RetrievalResult]:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError
