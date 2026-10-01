from typing import Optional

from core.models import Chunk
from rag_strategies.graph.entity_extractor import EntityExtractor, HeuristicEntityExtractor
from rag_strategies.graph.graph_store import GraphStore


def build_graph(chunks: list[Chunk], extractor: Optional[EntityExtractor] = None, store: Optional[GraphStore] = None) -> GraphStore:
    extractor = extractor or HeuristicEntityExtractor()
    store = store or GraphStore()
    store.clear()
    for chunk in chunks:
        entities = extractor.extract(chunk.text)
        names = []
        for entity in entities:
            store.add_entity(entity, chunk.id)
            names.append(entity.name)
        for index, left in enumerate(names):
            for right in names[index + 1:]:
                store.add_relation(left, right, "co_occurs")
    return store
