from core.chunking import RecursiveCharacterChunker
from core.config import HybridConfig, RetrievalConfig
from core.embeddings.hash_embedding import HashEmbedding
from core.evaluation import retrieval_overlap
from core.generation.extractive import ExtractiveGenerator
from core.models import Document, RetrievalResult, Chunk
from core.services.compare import compare_runs
from core.sparse_retrieval import BM25Retriever
from core.vectorstores import NumpyVectorStore
from rag_strategies.hybrid.pipeline import HybridRAG
from rag_strategies.naive.pipeline import NaiveRAG


def test_overlap_counts():
    a = Chunk("1", "one", "s")
    b = Chunk("2", "two", "s")
    c = Chunk("3", "three", "s")
    left = [RetrievalResult(a, 1, 1), RetrievalResult(b, 1, 2)]
    right = [RetrievalResult(b, 1, 1), RetrievalResult(c, 1, 2)]
    overlap = retrieval_overlap(left, right)
    assert overlap == {"both": 1, "a_only": 1, "b_only": 1, "a_total": 2, "b_total": 2}


def test_compare_runs_two_strategies():
    docs = [Document("Naive retrieval uses vectors. Hybrid retrieval also uses BM25 keywords.", "x.txt")]
    embedder = HashEmbedding()
    store = NumpyVectorStore()
    chunker = RecursiveCharacterChunker(200, 20)
    generator = ExtractiveGenerator()
    naive = NaiveRAG(chunker, embedder, store, generator, retrieval=RetrievalConfig(top_k=3, similarity_threshold=-1.0))
    naive.ingest(docs)
    hybrid = HybridRAG(
        chunker,
        embedder,
        store,
        BM25Retriever(),
        generator,
        retrieval=HybridConfig(dense_k=3, sparse_k=3, final_k=3),
    )
    hybrid.ingest(docs)
    experiment = compare_runs("What does hybrid retrieval use?", naive, hybrid, {"strategy": "naive"}, {"strategy": "hybrid"})
    assert experiment.pipeline_a.strategy == "naive"
    assert experiment.pipeline_b.strategy == "hybrid"
    assert "both" in experiment.overlap
    assert experiment.pipeline_a.question == experiment.pipeline_b.question
