from typing import Optional

from core.models import Chunk, RetrievalResult
from rag_strategies.graph.entity_extractor import EntityExtractor, HeuristicEntityExtractor
from rag_strategies.graph.graph_store import GraphStore


class GraphRetriever:
    def __init__(self, store: GraphStore, chunks: list[Chunk], extractor: Optional[EntityExtractor] = None):
        self.store = store
        self.extractor = extractor or HeuristicEntityExtractor()
        self.chunk_map = {chunk.id: chunk for chunk in chunks}

    def retrieve(self, query: str, top_k: int = 5, depth: int = 2) -> tuple[list[RetrievalResult], dict]:
        query_entities = self.extractor.extract(query)
        names = [entity.name for entity in query_entities]
        matched = self.store.match(names or query.split())
        visited = self.store.traverse(matched, depth)
        chunk_ids = self.store.chunk_ids_for(visited)
        results: list[RetrievalResult] = []
        for rank, chunk_id in enumerate(chunk_ids[:top_k], 1):
            chunk = self.chunk_map.get(chunk_id)
            if not chunk:
                continue
            hops = 0 if chunk_id in self.store.chunk_ids_for(matched) else 1
            results.append(RetrievalResult(chunk=chunk, score=1.0 / (1 + hops), rank=len(results) + 1, channel="graph"))
        extras = {
            "query_entities": [entity.name for entity in query_entities],
            "matched_nodes": matched,
            "visited_nodes": visited,
            "subgraph_edges": self.store.subgraph_edges(visited),
            "depth": depth,
        }
        return results, extras
