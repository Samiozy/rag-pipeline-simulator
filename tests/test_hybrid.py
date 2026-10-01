from core.chunking import FixedCharacterChunker
from core.config import HybridConfig
from core.embeddings.hash_embedding import HashEmbedding
from core.fusion import fuse
from core.generation.extractive import ExtractiveGenerator
from core.models import Chunk, RetrievalResult
from core.models import Document
from core.sparse_retrieval import BM25Retriever
from core.vectorstores import NumpyVectorStore
from rag_strategies.hybrid.pipeline import HybridRAG


def test_bm25_finds_keyword_chunk():
    chunks = [
        Chunk("1", "The sky is blue today.", "a"),
        Chunk("2", "Hybrid retrieval combines dense and sparse search.", "a"),
        Chunk("3", "Unrelated gardening notes.", "a"),
    ]
    retriever = BM25Retriever()
    retriever.index(chunks)
    results = retriever.search("hybrid sparse retrieval", k=2)
    assert results
    assert results[0].chunk.id == "2"
    assert results[0].channel == "sparse"


def test_rrf_and_weighted_fusion_handle_duplicates():
    chunk_a = Chunk("a", "alpha", "s")
    chunk_b = Chunk("b", "beta", "s")
    dense = [RetrievalResult(chunk_a, 0.9, 1), RetrievalResult(chunk_b, 0.2, 2)]
    sparse = [RetrievalResult(chunk_b, 5.0, 1), RetrievalResult(chunk_a, 1.0, 2)]
    rrf, rrf_rows = fuse(dense, sparse, method="rrf", final_k=2)
    weighted, _ = fuse(dense, sparse, method="weighted", final_k=2, dense_weight=0.6, sparse_weight=0.4)
    assert {item.chunk.id for item in rrf} == {"a", "b"}
    assert len(rrf) == 2
    assert {item.chunk.id for item in weighted} == {"a", "b"}
    assert {row.chunk_id for row in rrf_rows} == {"a", "b"}


def test_hybrid_run_exposes_fusion_table():
    docs = [Document(
        "Dense embeddings capture meaning. BM25 captures exact keywords such as BM25fusiontoken. "
        "A third sentence is about weather and gardens.",
        "lab.txt",
    )]
    strategy = HybridRAG(
        FixedCharacterChunker(80, 10),
        HashEmbedding(),
        NumpyVectorStore(),
        BM25Retriever(),
        ExtractiveGenerator(),
        retrieval=HybridConfig(dense_k=4, sparse_k=4, final_k=3, fusion_method="rrf"),
    )
    strategy.ingest(docs)
    results = strategy.retrieve("BM25fusiontoken keywords")
    assert results
    extras = strategy.last_extras
    assert extras["dense"] is not None
    assert extras["sparse"]
    assert extras["fusion_rows"]
    assert extras["sparse"][0].chunk.text.lower().find("bm25fusiontoken") >= 0
