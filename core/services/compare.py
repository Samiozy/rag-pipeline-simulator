from typing import Any

from core.evaluation import retrieval_overlap
from core.models import ExperimentResult, GenerationResult
from rag_strategies.base import RAGStrategy


def compare_runs(
    query: str,
    strategy_a: RAGStrategy,
    strategy_b: RAGStrategy,
    config_a: dict[str, Any],
    config_b: dict[str, Any],
    **run_kwargs,
) -> ExperimentResult:
    result_a: GenerationResult = strategy_a.run(query, **run_kwargs)
    result_b: GenerationResult = strategy_b.run(query, **run_kwargs)
    overlap = retrieval_overlap(result_a.retrieved, result_b.retrieved)
    return ExperimentResult(
        query=query,
        pipeline_a=result_a,
        pipeline_b=result_b,
        overlap=overlap,
        config_a=config_a,
        config_b=config_b,
    )
