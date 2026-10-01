from typing import Callable

from rag_strategies.agentic.multi_agent.pipeline import MultiAgentRAG
from rag_strategies.agentic.router.pipeline import RouterRAG
from rag_strategies.base import RAGStrategy
from rag_strategies.graph.pipeline import GraphRAG
from rag_strategies.hybrid.pipeline import HybridRAG
from rag_strategies.multimodal.pipeline import MultimodalRAG
from rag_strategies.naive.pipeline import NaiveRAG
from rag_strategies.rerank.pipeline import RerankRAG

STRATEGIES: dict[str, type[RAGStrategy]] = {
    "naive": NaiveRAG,
    "rerank": RerankRAG,
    "hybrid": HybridRAG,
    "graph": GraphRAG,
    "multimodal": MultimodalRAG,
    "agentic_router": RouterRAG,
    "multi_agent": MultiAgentRAG,
}

_LABELS = {
    "naive": "Naive RAG",
    "rerank": "Retrieve-and-Rerank RAG",
    "hybrid": "Hybrid RAG",
    "graph": "Graph RAG",
    "multimodal": "Multimodal RAG",
    "agentic_router": "Agentic RAG — Router",
    "multi_agent": "Multi-Agent RAG",
}


def available_strategies() -> list[str]:
    return list(STRATEGIES.keys())


def strategy_labels() -> dict[str, str]:
    return dict(_LABELS)


def create_strategy(key: str, **kwargs) -> RAGStrategy:
    try:
        cls: Callable[..., RAGStrategy] = STRATEGIES[key]
    except KeyError as exc:
        raise KeyError(f"Unknown RAG strategy: {key}") from exc
    return cls(**kwargs)
