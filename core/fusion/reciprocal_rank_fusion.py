from collections import defaultdict

from core.models import RetrievalResult


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievalResult]],
    k: int = 60,
) -> list[tuple[str, float, dict]]:
    """Return (chunk_id, fused_score, payload) sorted by fused score."""
    fused: dict[str, float] = defaultdict(float)
    payload: dict[str, dict] = {}
    for results in result_lists:
        for result in results:
            chunk_id = result.chunk.id
            fused[chunk_id] += 1.0 / (k + result.rank)
            payload.setdefault(chunk_id, {"chunk": result.chunk})
    ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)
    return [(chunk_id, score, payload[chunk_id]) for chunk_id, score in ranked]
