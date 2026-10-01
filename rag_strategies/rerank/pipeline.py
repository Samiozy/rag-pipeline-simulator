from typing import Optional

from core.chunking.base import Chunker
from core.config import RerankConfig
from core.embeddings.base import EmbeddingProvider
from core.generation.base import Generator
from core.models import Chunk, Document, RetrievalResult
from core.prompts import build_retrieval_query
from core.reranking.base import Reranker
from core.reranking.identity import IdentityReranker
from core.tracing import Tracer
from core.vectorstores.base import VectorStore
from rag_strategies.base import RAGStrategy


class RerankRAG(RAGStrategy):
    """Retrieve a wider candidate set, then reorder with a reranker."""

    key = "rerank"
    label = "Retrieve-and-Rerank RAG"
    description = "A second relevance model reorders the first-pass vector hits."

    def __init__(
        self,
        chunker: Chunker,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        generator: Generator,
        reranker: Optional[Reranker] = None,
        retrieval: Optional[RerankConfig] = None,
        tracer: Optional[Tracer] = None,
    ):
        super().__init__(tracer=tracer)
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.generator = generator
        self.reranker = reranker or IdentityReranker()
        self.retrieval_config = retrieval or RerankConfig()
        self.last_extras: dict = {}

    def ingest(self, documents: list[Document]) -> list[Chunk]:
        with self.tracer.span("chunking", input_summary=f"{len(documents)} documents") as span:
            self.chunks = self.chunker.chunk(documents)
            span["output_summary"] = f"{len(self.chunks)} chunks"
        with self.tracer.span("embedding", input_summary=f"{len(self.chunks)} chunks") as span:
            embeddings = self.embedder.embed_documents([chunk.text for chunk in self.chunks])
            span["output_summary"] = f"{len(self.chunks)} embeddings"
        with self.tracer.span("indexing", input_summary="dense vector store") as span:
            self.vector_store.clear()
            self.vector_store.add(self.chunks, embeddings)
            span["output_summary"] = f"{len(self.chunks)} vectors stored"
        return self.chunks

    def retrieve(self, query: str, **kwargs) -> list[RetrievalResult]:
        candidate_k = kwargs.get("candidate_k", self.retrieval_config.candidate_k)
        top_k = kwargs.get("top_k", self.retrieval_config.top_k)
        threshold = kwargs.get("threshold", self.retrieval_config.similarity_threshold or 0.0)
        history = kwargs.get("history")
        retrieval_query = build_retrieval_query(query, history)
        with self.tracer.span("query_embedding", input_summary=retrieval_query[:80]) as span:
            query_embedding = self.embedder.embed_query(retrieval_query)
            span["output_summary"] = "query vector"
        with self.tracer.span("dense_retrieval", input_summary=f"candidate_k={candidate_k}") as span:
            candidates = self.vector_store.search(query_embedding, k=candidate_k, threshold=threshold)
            span["output_summary"] = f"{len(candidates)} candidates"
        with self.tracer.span("reranking", input_summary=f"top_k={top_k}") as span:
            reranked = self.reranker.rerank(retrieval_query, candidates, top_k=top_k)
            span["output_summary"] = f"{len(reranked)} reranked chunks"
        self.last_extras = {
            "retrieval_query": retrieval_query,
            "candidates": candidates,
            "reranked": reranked,
        }
        return reranked
