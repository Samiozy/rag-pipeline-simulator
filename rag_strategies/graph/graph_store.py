from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from rag_strategies.graph.entity_extractor import Entity


@dataclass
class GraphNode:
    name: str
    entity_type: str
    chunk_ids: list[str] = field(default_factory=list)


@dataclass
class GraphEdge:
    source: str
    target: str
    relation: str


class GraphStore:
    """Local knowledge graph. Uses NetworkX when installed, otherwise an adjacency map."""

    def __init__(self) -> None:
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[GraphEdge] = []
        self._adj: dict[str, set[str]] = {}
        self._nx_graph = None
        try:
            import networkx as nx
            self._nx_graph = nx.Graph()
        except ImportError:
            self._nx_graph = None

    def clear(self) -> None:
        self.nodes.clear()
        self.edges.clear()
        self._adj.clear()
        if self._nx_graph is not None:
            self._nx_graph.clear()

    def add_entity(self, entity: Entity, chunk_id: str) -> None:
        key = entity.name.lower()
        node = self.nodes.get(key)
        if node is None:
            node = GraphNode(name=entity.name, entity_type=entity.entity_type, chunk_ids=[chunk_id])
            self.nodes[key] = node
            if self._nx_graph is not None:
                self._nx_graph.add_node(key, label=entity.name, entity_type=entity.entity_type)
        elif chunk_id not in node.chunk_ids:
            node.chunk_ids.append(chunk_id)

    def add_relation(self, left: str, right: str, relation: str) -> None:
        a, b = left.lower(), right.lower()
        if a == b or a not in self.nodes or b not in self.nodes:
            return
        pair = tuple(sorted((a, b)))
        if any(tuple(sorted((edge.source, edge.target))) == pair and edge.relation == relation for edge in self.edges):
            return
        self.edges.append(GraphEdge(source=a, target=b, relation=relation))
        self._adj.setdefault(a, set()).add(b)
        self._adj.setdefault(b, set()).add(a)
        if self._nx_graph is not None:
            self._nx_graph.add_edge(a, b, relation=relation)

    def match(self, names: Iterable[str]) -> list[str]:
        matched = []
        needles = [name.lower() for name in names if name]
        for key, node in self.nodes.items():
            if any(needle == key or needle in key or key in needle for needle in needles):
                matched.append(node.name.lower())
        return list(dict.fromkeys(matched))

    def traverse(self, seeds: list[str], depth: int) -> list[str]:
        frontier = {seed.lower() for seed in seeds if seed.lower() in self.nodes}
        seen = set(frontier)
        for _ in range(max(0, depth)):
            nxt = set()
            for node in frontier:
                nxt.update(self._adj.get(node, set()))
            nxt -= seen
            seen.update(nxt)
            frontier = nxt
            if not frontier:
                break
        return list(seen)

    def chunk_ids_for(self, node_keys: Iterable[str]) -> list[str]:
        ids: list[str] = []
        for key in node_keys:
            node = self.nodes.get(key.lower())
            if node:
                ids.extend(node.chunk_ids)
        return list(dict.fromkeys(ids))

    def subgraph_edges(self, node_keys: Iterable[str]) -> list[GraphEdge]:
        allowed = {key.lower() for key in node_keys}
        return [edge for edge in self.edges if edge.source in allowed and edge.target in allowed]

    def stats(self) -> dict[str, Any]:
        return {"nodes": len(self.nodes), "edges": len(self.edges), "backend": "networkx" if self._nx_graph is not None else "adjacency"}
