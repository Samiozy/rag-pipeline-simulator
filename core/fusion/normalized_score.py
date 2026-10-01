from collections import defaultdict

from core.models import RetrievalResult


def _minmax(results: list[RetrievalResult]) -> dict[str, float]:
    if not results:
        return {}
    scores = [result.score for result in results]
    low, high = min(scores), max(scores)
    span = high - low
    if span <= 1e-12:
        return {result.chunk.id: 1.0 for result in results}
    return {result.chunk.id: (result.score - low) / span for result in results}


def normalized_score_fusion(
    dense: list[RetrievalResult],
    sparse: list[RetrievalResult],
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
) -> list[tuple[str, float, dict]]:
    dense_norm = _minmax(dense)
    sparse_norm = _minmax(sparse)
    payload: dict[str, dict] = {}
    fused: dict[str, float] = defaultdict(float)
    for result in dense:
        payload.setdefault(result.chunk.id, {"chunk": result.chunk})
        fused[result.chunk.id] += dense_weight * dense_norm.get(result.chunk.id, 0.0)
    for result in sparse:
        payload.setdefault(result.chunk.id, {"chunk": result.chunk})
        fused[result.chunk.id] += sparse_weight * sparse_norm.get(result.chunk.id, 0.0)
    ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)
    return [(chunk_id, score, payload[chunk_id]) for chunk_id, score in ranked]
