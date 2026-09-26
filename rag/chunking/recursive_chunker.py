import uuid
from typing import Optional
from models import Document, Chunk
from .base import Chunker


class RecursiveCharacterChunker(Chunker):
    """Dependency-light recursive splitter that prefers natural boundaries."""

    def __init__(self, chunk_size: int = 800, overlap: int = 150, separators: Optional[list[str]] = None):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be >= 0 and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or ["\n\n", "\n", ". ", " "]

    def _split(self, text: str, separators: list[str]) -> list[str]:
        text = text.strip()
        if len(text) <= self.chunk_size:
            return [text] if text else []
        if not separators:
            return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        sep = separators[0]
        parts = text.split(sep)
        if len(parts) == 1:
            return self._split(text, separators[1:])

        output: list[str] = []
        current = ""
        for part in parts:
            candidate = (current + sep + part).strip() if current else part.strip()
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    output.append(current)
                if len(part) > self.chunk_size:
                    output.extend(self._split(part, separators[1:]))
                    current = ""
                else:
                    current = part.strip()
        if current:
            output.append(current)
        return output

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc_index, doc in enumerate(documents):
            base_parts = self._split(doc.text, self.separators)
            previous_tail = ""
            for chunk_index, part in enumerate(base_parts):
                text = (previous_tail + " " + part).strip() if previous_tail else part
                if len(text) > self.chunk_size:
                    text = text[-self.chunk_size:]
                metadata = dict(doc.metadata)
                metadata.update({"doc_index": doc_index, "chunk_index": chunk_index})
                chunks.append(Chunk(id=str(uuid.uuid4()), text=text, source=doc.source, metadata=metadata))
                previous_tail = part[-self.overlap:] if self.overlap else ""
        return chunks
