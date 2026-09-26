from .base import Chunker
from .fixed_chunker import FixedCharacterChunker
from .recursive_chunker import RecursiveCharacterChunker
from .sentence_chunker import SentenceChunker


def make_chunker(name: str, chunk_size: int, overlap: int) -> Chunker:
    if name == "Fixed Character":
        return FixedCharacterChunker(chunk_size, overlap)
    if name == "Sentence":
        return SentenceChunker(chunk_size)
    return RecursiveCharacterChunker(chunk_size, overlap)


__all__ = ["Chunker", "FixedCharacterChunker", "RecursiveCharacterChunker", "SentenceChunker", "make_chunker"]
