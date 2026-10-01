import re
import uuid

from core.models import Chunk, Document
from .base import Chunker


class SentenceChunker(Chunker):
    def __init__(self, max_chars: int = 800):
        self.max_chars = max_chars

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        output: list[Chunk] = []
        for doc_index, doc in enumerate(documents):
            sentences = re.split(r"(?<=[.!?])\s+", doc.text.strip())
            current: list[str] = []
            current_len = 0
            chunk_index = 0
            for sentence in sentences:
                if current and current_len + len(sentence) + 1 > self.max_chars:
                    text = " ".join(current)
                    metadata = dict(doc.metadata)
                    metadata.update({"doc_index": doc_index, "chunk_index": chunk_index})
                    output.append(Chunk(str(uuid.uuid4()), text, doc.source, metadata))
                    chunk_index += 1
                    current, current_len = [], 0
                current.append(sentence)
                current_len += len(sentence) + 1
            if current:
                metadata = dict(doc.metadata)
                metadata.update({"doc_index": doc_index, "chunk_index": chunk_index})
                output.append(Chunk(str(uuid.uuid4()), " ".join(current), doc.source, metadata))
        return output
