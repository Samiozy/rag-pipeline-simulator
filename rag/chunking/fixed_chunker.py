import uuid
from models import Document, Chunk
from .base import Chunker


class FixedCharacterChunker(Chunker):
    def __init__(self, chunk_size: int = 800, overlap: int = 150):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be >= 0 and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        step = self.chunk_size - self.overlap
        for doc_index, doc in enumerate(documents):
            text = doc.text
            for start in range(0, len(text), step):
                piece = text[start : start + self.chunk_size].strip()
                if not piece:
                    continue
                metadata = dict(doc.metadata)
                metadata.update({"doc_index": doc_index, "char_start": start, "char_end": start + len(piece)})
                chunks.append(Chunk(id=str(uuid.uuid4()), text=piece, source=doc.source, metadata=metadata))
                if start + self.chunk_size >= len(text):
                    break
        return chunks
