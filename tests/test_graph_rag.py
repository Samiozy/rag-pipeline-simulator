from core.chunking import RecursiveCharacterChunker
from core.config import GraphConfig
from core.generation.extractive import ExtractiveGenerator
from core.models import Document
from rag_strategies.graph.entity_extractor import HeuristicEntityExtractor
from rag_strategies.graph.pipeline import GraphRAG


def test_heuristic_extractor_finds_concepts_and_names():
    entities = HeuristicEntityExtractor().extract(
        "Retrieval-Augmented Generation uses embeddings. Chunk overlap helps Retrieval."
    )
    names = {entity.name.lower() for entity in entities}
    assert "embeddings" in names or "embedding" in names
    assert any("retrieval" in name for name in names)


def test_graph_rag_retrieves_related_chunks():
    docs = [Document(
        "Retrieval-Augmented Generation combines retrieval with generation. "
        "Chunk overlap is used at chunk boundaries. "
        "The generator should be evaluated separately from retrieval.",
        "sample.txt",
    )]
    strategy = GraphRAG(
        RecursiveCharacterChunker(220, 30),
        ExtractiveGenerator(),
        retrieval=GraphConfig(top_k=3, traversal_depth=2),
    )
    chunks = strategy.ingest(docs)
    assert chunks
    assert strategy.graph_store.nodes
    results = strategy.retrieve("How does chunk overlap affect retrieval?")
    assert results
    extras = strategy.last_extras
    assert extras["matched_nodes"] or extras["query_entities"]
    output = strategy.run("Why evaluate retrieval independently from generation?")
    assert output.strategy == "graph"
    assert output.answer
