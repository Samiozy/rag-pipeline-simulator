from core.chunking import FixedCharacterChunker
from core.config import RerankConfig
from core.embeddings.hash_embedding import HashEmbedding
from core.generation.extractive import ExtractiveGenerator
from core.models import Document
from core.reranking import KeywordReranker
from core.vectorstores import NumpyVectorStore
from rag_strategies.rerank.pipeline import RerankRAG


def test_rerank_keeps_final_top_k():
    text = (
        "Cats sit on mats. " * 8
        + "Hybrid retrieval combines dense and sparse search. " * 8
        + "Dogs chase balls. " * 8
    )
    docs = [Document(text, "notes.txt")]
    strategy = RerankRAG(
        FixedCharacterChunker(120, 20),
        HashEmbedding(),
        NumpyVectorStore(),
        ExtractiveGenerator(),
        reranker=KeywordReranker(),
        retrieval=RerankConfig(candidate_k=8, top_k=3, similarity_threshold=-1.0),
    )
    strategy.ingest(docs)
    results = strategy.retrieve("hybrid retrieval dense sparse")
    assert len(results) <= 3
    extras = strategy.last_extras
    assert extras["candidates"]
    assert len(extras["candidates"]) >= len(results)
    assert extras["reranked"] == results
    for result in results:
        assert result.original_rank is not None
