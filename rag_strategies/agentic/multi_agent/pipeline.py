from typing import Optional

from core.config import MultiAgentConfig
from core.generation.base import Generator
from core.models import Chunk, Document, RetrievalResult
from core.tracing import Tracer
from rag_strategies.base import RAGStrategy
from rag_strategies.agentic.multi_agent.agents import DocumentAgent, GraphAgent, SearchAgent
from rag_strategies.agentic.multi_agent.coordinator import Coordinator
from rag_strategies.agentic.multi_agent.message import AgentMessage
from rag_strategies.graph.graph_store import GraphStore
from rag_strategies.naive.pipeline import NaiveRAG


class MultiAgentRAG(RAGStrategy):
    """Retrieval tasks are delegated to specialized agents and then synthesized."""

    key = "multi_agent"
    label = "Multi-Agent RAG"
    description = "A coordinator delegates document, lexical, and graph retrieval, then synthesizes."

    def __init__(
        self,
        document_strategy: NaiveRAG,
        generator: Generator,
        sparse_retriever=None,
        graph_store: Optional[GraphStore] = None,
        retrieval: Optional[MultiAgentConfig] = None,
        tracer: Optional[Tracer] = None,
    ):
        super().__init__(tracer=tracer)
        self.document_strategy = document_strategy
        self.generator = generator
        self.sparse_retriever = sparse_retriever
        self.graph_store = graph_store or GraphStore()
        self.retrieval_config = retrieval or MultiAgentConfig()
        self.last_extras: dict = {}

    def ingest(self, documents: list[Document]) -> list[Chunk]:
        chunks = self.document_strategy.ingest(documents)
        self.chunks = chunks
        return chunks

    def retrieve(self, query: str, **kwargs) -> list[RetrievalResult]:
        top_k = kwargs.get("top_k", self.retrieval_config.top_k)
        document_agent = DocumentAgent(self.document_strategy)
        search_agent = SearchAgent(self.sparse_retriever) if self.sparse_retriever and self.retrieval_config.enable_search_agent else None
        graph_agent = (
            GraphAgent(self.graph_store, self.chunks or self.document_strategy.chunks)
            if self.retrieval_config.enable_graph_agent
            else None
        )
        coordinator = Coordinator(document_agent, search_agent, graph_agent)
        with self.tracer.span("routing", input_summary="coordinator delegation") as span:
            results, messages = coordinator.delegate(query, top_k)
            span["output_summary"] = f"{len(messages)} messages, {len(results)} merged chunks"
        synthesis = AgentMessage("synthesis_agent", "user", "synthesis", f"Prepared {len(results)} chunks for the generator.")
        messages.append(synthesis)
        self.last_extras = {"retrieval_query": query, "messages": messages, "agents": ["coordinator", "document_agent", "search_agent", "graph_agent", "synthesis_agent"]}
        return results
