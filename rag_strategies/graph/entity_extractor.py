from abc import ABC, abstractmethod
from dataclasses import dataclass
import re

CONCEPTS = {
    "rag",
    "retrieval",
    "generation",
    "embedding",
    "embeddings",
    "chunk",
    "chunks",
    "chunking",
    "overlap",
    "vector",
    "index",
    "reranker",
    "reranking",
    "bm25",
    "hybrid",
    "prompt",
    "generator",
    "latency",
    "context",
    "similarity",
}

_TITLE_CASE = re.compile(r"\b([A-Z][A-Za-z0-9+-]+(?:\s+[A-Z][A-Za-z0-9+-]+)+)\b")
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9+-]{2,}")


@dataclass
class Entity:
    name: str
    entity_type: str


class EntityExtractor(ABC):
    @abstractmethod
    def extract(self, text: str) -> list[Entity]:
        raise NotImplementedError


class HeuristicEntityExtractor(EntityExtractor):
    """Offline extractor. Finds title-case phrases and a small concept lexicon."""

    def extract(self, text: str) -> list[Entity]:
        found: dict[str, Entity] = {}
        for match in _TITLE_CASE.finditer(text):
            name = re.sub(r"\s+", " ", match.group(1)).strip()
            key = name.lower()
            found[key] = Entity(name=name, entity_type="named")
        for token in _TOKEN.findall(text):
            key = token.lower()
            if key in CONCEPTS and key not in found:
                found[key] = Entity(name=key, entity_type="concept")
        return list(found.values())
