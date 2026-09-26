from abc import ABC, abstractmethod
from models import Document, Chunk


class Chunker(ABC):
    @abstractmethod
    def chunk(self, documents: list[Document]) -> list[Chunk]:
        raise NotImplementedError
