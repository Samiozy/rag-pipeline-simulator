from abc import ABC, abstractmethod
from typing import Optional
from models import RetrievalResult


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, results: list[RetrievalResult], top_n: Optional[int] = None) -> list[RetrievalResult]:
        raise NotImplementedError


class NoOpReranker(Reranker):
    def rerank(self, query: str, results: list[RetrievalResult], top_n: Optional[int] = None) -> list[RetrievalResult]:
        selected = results[:top_n] if top_n else results
        for rank, result in enumerate(selected, 1):
            result.rank = rank
        return selected


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, results: list[RetrievalResult], top_n: Optional[int] = None) -> list[RetrievalResult]:
        if not results:
            return []
        pairs = [(query, r.chunk.text) for r in results]
        scores = self.model.predict(pairs)
        reranked = sorted(zip(results, scores), key=lambda x: float(x[1]), reverse=True)
        selected = reranked[:top_n] if top_n else reranked
        output: list[RetrievalResult] = []
        for rank, (result, score) in enumerate(selected, 1):
            output.append(RetrievalResult(result.chunk, float(score), rank))
        return output
