from models import Document
from rag.chunking.fixed_chunker import FixedCharacterChunker
from rag.chunking.recursive_chunker import RecursiveCharacterChunker


def test_fixed_chunker_creates_chunks():
    docs = [Document("a" * 1000, "test.txt")]
    chunks = FixedCharacterChunker(chunk_size=300, overlap=50).chunk(docs)
    assert len(chunks) >= 4
    assert all(len(c.text) <= 300 for c in chunks)


def test_recursive_chunker_preserves_source():
    docs = [Document("First paragraph.\n\nSecond paragraph. " * 20, "source.txt")]
    chunks = RecursiveCharacterChunker(chunk_size=200, overlap=20).chunk(docs)
    assert chunks
    assert all(c.source == "source.txt" for c in chunks)
