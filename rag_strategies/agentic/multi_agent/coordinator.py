from typing import Optional

from core.models import RetrievalResult
from rag_strategies.agentic.multi_agent.agents import DocumentAgent, GraphAgent, SearchAgent
from rag_strategies.agentic.multi_agent.message import AgentMessage


class Coordinator:
    """Delegates retrieval, then returns merged evidence. No external writes."""

    def __init__(self, document_agent: DocumentAgent, search_agent: Optional[SearchAgent] = None, graph_agent: Optional[GraphAgent] = None):
        self.document_agent = document_agent
        self.search_agent = search_agent
        self.graph_agent = graph_agent

    def delegate(self, query: str, top_k: int) -> tuple[list[RetrievalResult], list[AgentMessage]]:
        messages = [
            AgentMessage("coordinator", "document_agent", "delegate", f"Retrieve document evidence for: {query}"),
        ]
        merged: dict[str, RetrievalResult] = {}
        doc_results, doc_msg = self.document_agent.retrieve(query, top_k)
        messages.append(doc_msg)
        for result in doc_results:
            merged[result.chunk.id] = result
        if self.search_agent:
            messages.append(AgentMessage("coordinator", "search_agent", "delegate", "Run lexical search."))
            search_results, search_msg = self.search_agent.retrieve(query, top_k)
            messages.append(search_msg)
            for result in search_results:
                merged.setdefault(result.chunk.id, result)
        if self.graph_agent:
            messages.append(AgentMessage("coordinator", "graph_agent", "delegate", "Walk the entity graph."))
            graph_results, graph_msg = self.graph_agent.retrieve(query, top_k)
            messages.append(graph_msg)
            for result in graph_results:
                merged.setdefault(result.chunk.id, result)
        ranked = []
        for rank, result in enumerate(merged.values(), 1):
            result.rank = rank
            ranked.append(result)
        messages.append(AgentMessage("coordinator", "synthesis_agent", "delegate", f"Synthesize {len(ranked)} unique chunks."))
        return ranked[:top_k], messages
