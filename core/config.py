from dataclasses import dataclass
from typing import Optional


@dataclass
class RetrievalConfig:
    top_k: int = 5
    similarity_threshold: Optional[float] = 0.0


@dataclass
class RerankConfig:
    candidate_k: int = 20
    top_k: int = 5
    similarity_threshold: Optional[float] = 0.0


@dataclass
class HybridConfig:
    dense_k: int = 10
    sparse_k: int = 10
    final_k: int = 5
    fusion_method: str = "rrf"
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    rrf_k: int = 60


@dataclass
class GraphConfig:
    top_k: int = 5
    traversal_depth: int = 2


@dataclass
class MultimodalConfig:
    top_k: int = 5
    similarity_threshold: Optional[float] = 0.0


@dataclass
class RouterConfig:
    default_route: str = "rag"
    top_k: int = 5


@dataclass
class MultiAgentConfig:
    top_k: int = 5
    enable_graph_agent: bool = True
    enable_search_agent: bool = True


@dataclass
class GenerationConfig:
    temperature: float = 0.2
    max_tokens: int = 700
    system_prompt: Optional[str] = None
