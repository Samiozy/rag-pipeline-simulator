from typing import Union
from models import RetrievalResult


def retrieval_summary(results: list[RetrievalResult]) -> dict[str, Union[float, int]]:
    if not results:
        return {"retrieved": 0, "avg_score": 0.0, "max_score": 0.0, "min_score": 0.0}
    scores = [r.score for r in results]
    return {
        "retrieved": len(results),
        "avg_score": sum(scores) / len(scores),
        "max_score": max(scores),
        "min_score": min(scores),
    }
