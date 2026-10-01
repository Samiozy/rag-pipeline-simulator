from typing import Union

from core.models import RetrievalResult


def retrieval_summary(results: list[RetrievalResult]) -> dict[str, Union[float, int]]:
    if not results:
        return {"retrieved": 0, "avg_score": 0.0, "max_score": 0.0, "min_score": 0.0}
    scores = [result.score for result in results]
    return {
        "retrieved": len(results),
        "avg_score": sum(scores) / len(scores),
        "max_score": max(scores),
        "min_score": min(scores),
    }


def retrieval_overlap(results_a: list[RetrievalResult], results_b: list[RetrievalResult]) -> dict[str, int]:
    ids_a = {result.chunk.id for result in results_a}
    ids_b = {result.chunk.id for result in results_b}
    return {
        "both": len(ids_a & ids_b),
        "a_only": len(ids_a - ids_b),
        "b_only": len(ids_b - ids_a),
        "a_total": len(ids_a),
        "b_total": len(ids_b),
    }
