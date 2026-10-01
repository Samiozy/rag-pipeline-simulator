from dataclasses import dataclass
from typing import Optional

from core.models import RetrievalResult

from .normalized_score import normalized_score_fusion
from .reciprocal_rank_fusion import reciprocal_rank_fusion
from .weighted_score import weighted_score_fusion


@dataclass
class FusionRow:
    chunk_id: str
    preview: str
    source: str
    dense_rank: Optional[int]
    dense_score: Optional[float]
    sparse_rank: Optional[int]
    sparse_score: Optional[float]
    fused_rank: int
    fused_score: float


def _index_channel(results: list[RetrievalResult]) -> dict[str, RetrievalResult]:
    return {result.chunk.id: result for result in results}


def fuse(
    dense: list[RetrievalResult],
    sparse: list[RetrievalResult],
    method: str = "rrf",
    final_k: int = 5,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
    rrf_k: int = 60,
) -> tuple[list[RetrievalResult], list[FusionRow]]:
    if method == "weighted":
        ranked = weighted_score_fusion(dense, sparse, dense_weight, sparse_weight)
    elif method == "normalized":
        ranked = normalized_score_fusion(dense, sparse, 0.5, 0.5)
    else:
        ranked = reciprocal_rank_fusion([dense, sparse], k=rrf_k)

    dense_map = _index_channel(dense)
    sparse_map = _index_channel(sparse)
    fused_results: list[RetrievalResult] = []
    rows: list[FusionRow] = []
    for fused_rank, (chunk_id, fused_score, payload) in enumerate(ranked, 1):
        chunk = payload["chunk"]
        dense_hit = dense_map.get(chunk_id)
        sparse_hit = sparse_map.get(chunk_id)
        row = FusionRow(
            chunk_id=chunk_id,
            preview=chunk.text[:180].replace("\n", " "),
            source=chunk.source,
            dense_rank=dense_hit.rank if dense_hit else None,
            dense_score=dense_hit.score if dense_hit else None,
            sparse_rank=sparse_hit.rank if sparse_hit else None,
            sparse_score=sparse_hit.score if sparse_hit else None,
            fused_rank=fused_rank,
            fused_score=fused_score,
        )
        rows.append(row)
        if fused_rank <= final_k:
            fused_results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=float(fused_score),
                    rank=fused_rank,
                    channel="fused",
                    metadata={
                        "dense_rank": row.dense_rank,
                        "sparse_rank": row.sparse_rank,
                    },
                )
            )
    return fused_results, rows
