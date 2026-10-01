from typing import Optional

from core.chunking.base import Chunker
from core.config import HybridConfig
from core.embeddings.base import EmbeddingProvider
from core.fusion import fuse
from core.generation.base import Generator
from core.models import Chunk, Document, RetrievalResult
from core.prompts import build_retrieval_query
from core.sparse_retrieval.base import SparseRetriever
from core.tracing import Tracer
from core.vectorstores.base import VectorStore
from rag_strategies.base import RAGStrategy


class HybridRAG(RAGStrategy):
    """Combine dense vector search with BM25, then fuse rankings."""

    key = "hybrid"
    label = "Hybrid RAG"
    description = "Keyword retrieval is combined with semantic retrieval."

    def __init__(
        self,
        chunker: Chunker,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        sparse_retriever: SparseRetriever,
        generator,
        retrieval: Optional[HybridConfig] = None,
        tracer: Optional[Tracer] = None,
    ):
        super().__init__(tracer=tracer)
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.sparse_retriever = sparse_retriever
        self.generator = generator
        self.retrieval_config = retrieval or HybridConfig()
        self.last_extras: dict = {}

    def ingest(self, documents: list[Document]) -> list[Chunk]:
        with self.tracer.span("chunking", input_summary=f"{len(documents)} documents") as span:
            self.chunks = self.chunker.chunk(documents)
            span["output_summary"] = f"{len(self.chunks)} chunks"
        with self.tracer.span("embedding", input_summary=f"{len(self.chunks)} chunks") as span:
            embeddings = self.embedder.embed_documents([chunk.text for chunk in self.chunks])
            span["output_summary"] = f"{len(self.chunks)} embeddings"
        with self.tracer.span("indexing", input_summary="dense + BM25") as span:
            self.vector_store.clear()
            self.vector_store.add(self.chunks, embeddings)
            self.sparse_retriever.index(self.chunks)
            span["output_summary"] = f"{len(self.chunks)} dense vectors and BM25 documents"
        return self.chunks

    def retrieve(self, query: str, **kwargs) -> list[RetrievalResult]:
        cfg = self.retrieval_config
        dense_k = kwargs.get("dense_k", cfg.dense_k)
        sparse_k = kwargs.get("sparse_k", cfg.sparse_k)
        final_k = kwargs.get("final_k", cfg.final_k)
        method = kwargs.get("fusion_method", cfg.fusion_method)
        dense_weight = kwargs.get("dense_weight", cfg.dense_weight)
        sparse_weight = kwargs.get("sparse_weight", cfg.sparse_weight)
        history = kwargs.get("history")
        retrieval_query = build_retrieval_query(query, history)
        with self.tracer.span("query_embedding", input_summary=retrieval_query[:80]) as span:
            query_embedding = self.embedder.embed_query(retrieval_query)
            span["output_summary"] = "query vector"
        with self.tracer.span("dense_retrieval", input_summary=f"dense_k={dense_k}") as span:
            dense = self.vector_store.search(query_embedding, k=dense_k, threshold=0.0)
            span["output_summary"] = f"{len(dense)} dense hits"
        with self.tracer.span("sparse_retrieval", input_summary=f"sparse_k={sparse_k}") as span:
            sparse = self.sparse_retriever.search(retrieval_query, k=sparse_k)
            span["output_summary"] = f"{len(sparse)} BM25 hits"
        with self.tracer.span("fusion", input_summary=method) as span:
            fused, rows = fuse(
                dense,
                sparse,
                method=method,
                final_k=final_k,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
            )
            span["output_summary"] = f"{len(fused)} fused chunks"
        self.last_extras = {
            "retrieval_query": retrieval_query,
            "dense": dense,
            "sparse": sparse,
            "fusion_rows": rows,
            "fusion_method": method,
        }
        return fused
