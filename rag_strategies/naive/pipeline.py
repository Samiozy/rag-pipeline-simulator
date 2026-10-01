from typing import Optional

from core.chunking.base import Chunker
from core.config import RetrievalConfig
from core.embeddings.base import EmbeddingProvider
from core.generation.base import Generator
from core.models import Chunk, Document, RetrievalResult
from core.prompts import build_retrieval_query
from core.tracing import Tracer
from core.vectorstores.base import VectorStore
from rag_strategies.base import RAGStrategy


class NaiveRAG(RAGStrategy):
    """Embed documents, retrieve nearest chunks, then generate."""

    key = "naive"
    label = "Naive RAG"
    description = "The baseline: chunk, embed, vector search, prompt, generate."

    def __init__(
        self,
        chunker: Chunker,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        generator: Generator,
        retrieval: Optional[RetrievalConfig] = None,
        tracer: Optional[Tracer] = None,
    ):
        super().__init__(tracer=tracer)
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.generator = generator
        self.retrieval_config = retrieval or RetrievalConfig()
        self.last_extras: dict = {}

    def ingest(self, documents: list[Document]) -> list[Chunk]:
        with self.tracer.span("chunking", input_summary=f"{len(documents)} documents") as span:
            self.chunks = self.chunker.chunk(documents)
            span["output_summary"] = f"{len(self.chunks)} chunks"
        with self.tracer.span("embedding", input_summary=f"{len(self.chunks)} chunks") as span:
            embeddings = self.embedder.embed_documents([chunk.text for chunk in self.chunks])
            span["output_summary"] = f"{len(self.chunks)} embeddings"
            span["metadata"] = {"shape": list(embeddings.shape)}
        with self.tracer.span("indexing", input_summary="dense vector store") as span:
            self.vector_store.clear()
            self.vector_store.add(self.chunks, embeddings)
            span["output_summary"] = f"{len(self.chunks)} vectors stored"
        return self.chunks

    def retrieve(self, query: str, **kwargs) -> list[RetrievalResult]:
        top_k = kwargs.get("top_k", self.retrieval_config.top_k)
        threshold = kwargs.get("threshold", self.retrieval_config.similarity_threshold or 0.0)
        history = kwargs.get("history")
        retrieval_query = build_retrieval_query(query, history)
        with self.tracer.span("query_embedding", input_summary=retrieval_query[:80]) as span:
            query_embedding = self.embedder.embed_query(retrieval_query)
            span["output_summary"] = "query vector"
        with self.tracer.span("dense_retrieval", input_summary=f"top_k={top_k}") as span:
            results = self.vector_store.search(query_embedding, k=top_k, threshold=threshold)
            span["output_summary"] = f"{len(results)} chunks retrieved"
        self.last_extras = {"retrieval_query": retrieval_query, "candidates": results}
        return results
