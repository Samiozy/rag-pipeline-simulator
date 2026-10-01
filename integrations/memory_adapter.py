from typing import Optional, Protocol

from core.models import ContextItem


class MemorySource(Protocol):
    def search(self, query: str, k: int = 5) -> list[ContextItem]:
        ...


class MemoryAdapter:
    """Placeholder for a later memory-simulator integration."""

    def __init__(self, source: Optional[MemorySource] = None):
        self.source = source

    def retrieve(self, query: str, k: int = 5) -> list[ContextItem]:
        if self.source is None:
            return []
        return self.source.search(query, k=k)
