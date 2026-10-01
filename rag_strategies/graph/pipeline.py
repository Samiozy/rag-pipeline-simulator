from typing import Optional

from core.chunking.base import Chunker
from core.config import GraphConfig
from core.generation.base import Generator
from core.models import Chunk, Document, RetrievalResult
from core.tracing import Tracer
from rag_strategies.base import RAGStrategy
from rag_strategies.graph.entity_extractor import HeuristicEntityExtractor
from rag_strategies.graph.graph_builder import build_graph
from rag_strategies.graph.graph_retriever import GraphRetriever
from rag_strategies.graph.graph_store import GraphStore


class GraphRAG(RAGStrategy):
    """Follow entity relationships instead of only vector similarity."""

    key = "graph"
    label = "Graph RAG"
    description = "Retrieval can follow relationships rather than only vector similarity."

    def __init__(
        self,
        chunker: Chunker,
        generator: Generator,
        graph_store: Optional[GraphStore] = None,
        retrieval: Optional[GraphConfig] = None,
        tracer: Optional[Tracer] = None,
    ):
        super().__init__(tracer=tracer)
        self.chunker = chunker
        self.generator = generator
        self.graph_store = graph_store or GraphStore()
        self.extractor = HeuristicEntityExtractor()
        self.retrieval_config = retrieval or GraphConfig()
        self.last_extras: dict = {}

    def ingest(self, documents: list[Document]) -> list[Chunk]:
        with self.tracer.span("chunking", input_summary=f"{len(documents)} documents") as span:
            self.chunks = self.chunker.chunk(documents)
            span["output_summary"] = f"{len(self.chunks)} chunks"
        with self.tracer.span("indexing", input_summary="knowledge graph") as span:
            build_graph(self.chunks, self.extractor, self.graph_store)
            span["output_summary"] = f"{len(self.graph_store.nodes)} nodes, {len(self.graph_store.edges)} edges"
        return self.chunks

    def retrieve(self, query: str, **kwargs) -> list[RetrievalResult]:
        top_k = kwargs.get("top_k", self.retrieval_config.top_k)
        depth = kwargs.get("traversal_depth", self.retrieval_config.traversal_depth)
        if not self.graph_store.nodes:
            raise ValueError("The graph has no extracted entities. Try a longer document or a different query.")
        with self.tracer.span("graph_retrieval", input_summary=query[:80]) as span:
            retriever = GraphRetriever(self.graph_store, self.chunks, self.extractor)
            results, extras = retriever.retrieve(query, top_k=top_k, depth=depth)
            span["output_summary"] = f"{len(results)} graph chunks, {len(extras['matched_nodes'])} matched entities"
        extras["retrieval_query"] = query
        extras["graph_stats"] = self.graph_store.stats()
        self.last_extras = extras
        return results
