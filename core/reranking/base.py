from abc import ABC, abstractmethod
from typing import Optional

from core.models import RetrievalResult


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, results: list[RetrievalResult], top_k: Optional[int] = None) -> list[RetrievalResult]:
        raise NotImplementedError
