from core.chunking import RecursiveCharacterChunker
from core.config import RetrievalConfig
from core.embeddings.hash_embedding import HashEmbedding
from core.generation.extractive import ExtractiveGenerator
from core.models import Document
from core.vectorstores import NumpyVectorStore
from rag_strategies.naive.pipeline import NaiveRAG


def _strategy() -> NaiveRAG:
    docs = [Document(
        "Retrieval-augmented generation combines retrieval with generation. "
        "Chunk size influences retrieval behavior. Overlap reduces boundary loss.",
        "sample.txt",
    )]
    strategy = NaiveRAG(
        RecursiveCharacterChunker(400, 40),
        HashEmbedding(),
        NumpyVectorStore(),
        ExtractiveGenerator(),
        retrieval=RetrievalConfig(top_k=3, similarity_threshold=-1.0),
    )
    strategy.ingest(docs)
    return strategy


def test_naive_ingest_and_retrieve():
    strategy = _strategy()
    assert strategy.chunks
    results = strategy.retrieve("How does chunk size influence retrieval?")
    assert results
    assert results[0].rank == 1


def test_naive_offline_run():
    strategy = _strategy()
    output = strategy.run("What is retrieval-augmented generation?")
    assert output.answer
    assert output.prompt
    assert output.strategy == "naive"
    assert output.traces
    assert output.context
