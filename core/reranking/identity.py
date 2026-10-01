from typing import Optional

from core.models import RetrievalResult
from .base import Reranker


class IdentityReranker(Reranker):
    """Leaves vector-search order unchanged."""

    def rerank(self, query: str, results: list[RetrievalResult], top_k: Optional[int] = None) -> list[RetrievalResult]:
        selected = results[:top_k] if top_k else list(results)
        output = []
        for rank, result in enumerate(selected, 1):
            output.append(
                RetrievalResult(
                    chunk=result.chunk,
                    score=result.score,
                    rank=rank,
                    channel="rerank",
                    original_rank=result.rank or rank,
                    metadata=dict(result.metadata),
                )
            )
        return output
