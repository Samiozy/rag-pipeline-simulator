from .normalized_score import normalized_score_fusion
from core.models import RetrievalResult


def weighted_score_fusion(
    dense: list[RetrievalResult],
    sparse: list[RetrievalResult],
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
) -> list[tuple[str, float, dict]]:
    """Weighted combination of min-max normalized channel scores."""
    return normalized_score_fusion(dense, sparse, dense_weight, sparse_weight)
