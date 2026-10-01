from abc import ABC, abstractmethod

from core.models import Chunk, RetrievalResult


class SparseRetriever(ABC):
    @abstractmethod
    def index(self, chunks: list[Chunk]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query: str, k: int) -> list[RetrievalResult]:
        raise NotImplementedError
