from typing import Optional

from core.models import RetrievalResult
from core.sparse_retrieval.bm25 import tokenize
from .base import Reranker


class KeywordReranker(Reranker):
    """Offline reranker for tests and demos. Scores by query-term overlap."""

    def rerank(self, query: str, results: list[RetrievalResult], top_k: Optional[int] = None) -> list[RetrievalResult]:
        terms = set(tokenize(query))
        scored = []
        for result in results:
            tokens = set(tokenize(result.chunk.text))
            score = float(len(terms & tokens))
            scored.append((score, result))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = scored[:top_k] if top_k else scored
        output = []
        for rank, (score, result) in enumerate(selected, 1):
            output.append(
                RetrievalResult(
                    chunk=result.chunk,
                    score=score,
                    rank=rank,
                    channel="rerank",
                    original_rank=result.rank,
                )
            )
        return output
