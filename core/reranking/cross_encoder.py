from typing import Optional

from core.models import RetrievalResult
from .base import Reranker


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError("Cross-encoder reranking requires sentence-transformers.") from exc
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, results: list[RetrievalResult], top_k: Optional[int] = None) -> list[RetrievalResult]:
        if not results:
            return []
        pairs = [(query, result.chunk.text) for result in results]
        scores = self.model.predict(pairs)
        reranked = sorted(zip(results, scores), key=lambda item: float(item[1]), reverse=True)
        selected = reranked[:top_k] if top_k else reranked
        output: list[RetrievalResult] = []
        for rank, (result, score) in enumerate(selected, 1):
            output.append(
                RetrievalResult(
                    chunk=result.chunk,
                    score=float(score),
                    rank=rank,
                    channel="rerank",
                    original_rank=result.rank,
                )
            )
        return output
