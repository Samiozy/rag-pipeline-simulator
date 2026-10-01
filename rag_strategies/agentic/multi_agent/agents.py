from core.models import RetrievalResult
from rag_strategies.agentic.multi_agent.message import AgentMessage
from rag_strategies.graph.graph_retriever import GraphRetriever
from rag_strategies.graph.graph_store import GraphStore
from rag_strategies.naive.pipeline import NaiveRAG


class DocumentAgent:
    def __init__(self, strategy: NaiveRAG):
        self.strategy = strategy

    def retrieve(self, query: str, top_k: int) -> tuple[list[RetrievalResult], AgentMessage]:
        try:
            results = self.strategy.retrieve(query, top_k=top_k)
            content = f"Retrieved {len(results)} document chunks."
        except Exception as exc:
            results = []
            content = f"Document agent failed: {exc}"
        return results, AgentMessage("document_agent", "coordinator", "retrieval", content, {"count": len(results)})


class SearchAgent:
    def __init__(self, sparse_retriever):
        self.sparse_retriever = sparse_retriever

    def retrieve(self, query: str, top_k: int) -> tuple[list[RetrievalResult], AgentMessage]:
        try:
            results = self.sparse_retriever.search(query, k=top_k)
            content = f"Lexical search returned {len(results)} chunks."
        except Exception as exc:
            results = []
            content = f"Search agent failed: {exc}"
        return results, AgentMessage("search_agent", "coordinator", "retrieval", content, {"count": len(results)})


class GraphAgent:
    def __init__(self, store: GraphStore, chunks):
        self.store = store
        self.chunks = chunks

    def retrieve(self, query: str, top_k: int) -> tuple[list[RetrievalResult], AgentMessage]:
        if not self.store.nodes:
            return [], AgentMessage("graph_agent", "coordinator", "retrieval", "Graph has no entities yet.", {"count": 0})
        try:
            results, _extras = GraphRetriever(self.store, self.chunks).retrieve(query, top_k=top_k)
            content = f"Graph walk returned {len(results)} chunks."
        except Exception as exc:
            results = []
            content = f"Graph agent failed: {exc}"
        return results, AgentMessage("graph_agent", "coordinator", "retrieval", content, {"count": len(results)})
