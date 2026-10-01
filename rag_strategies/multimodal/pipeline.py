from typing import Optional

from core.chunking.base import Chunker
from core.config import MultimodalConfig
from core.embeddings.base import EmbeddingProvider
from core.generation.base import Generator
from core.models import Chunk, Document, RetrievalResult
from core.prompts import build_retrieval_query
from core.tracing import Tracer
from core.vectorstores.base import VectorStore
from rag_strategies.base import RAGStrategy
from rag_strategies.multimodal.modality_router import group_by_modality, modality_of


class MultimodalRAG(RAGStrategy):
    """Retrieve mixed text and image-caption chunks, preserving modality."""

    key = "multimodal"
    label = "Multimodal RAG"
    description = "Text and image items share an index. Each hit records its modality."

    def __init__(
        self,
        chunker: Chunker,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        generator: Generator,
        retrieval: Optional[MultimodalConfig] = None,
        tracer: Optional[Tracer] = None,
    ):
        super().__init__(tracer=tracer)
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.generator = generator
        self.retrieval_config = retrieval or MultimodalConfig()
        self.last_extras: dict = {}

    def ingest(self, documents: list[Document]) -> list[Chunk]:
        for document in documents:
            document.metadata.setdefault("modality", "image" if document.metadata.get("file_type") in {"png", "jpg", "jpeg", "webp", "gif", "pdf-image"} else "text")
        with self.tracer.span("chunking", input_summary=f"{len(documents)} documents") as span:
            self.chunks = self.chunker.chunk(documents)
            for chunk in self.chunks:
                chunk.metadata.setdefault("modality", "text")
            span["output_summary"] = f"{len(self.chunks)} chunks"
        with self.tracer.span("embedding", input_summary="text + image captions") as span:
            embeddings = self.embedder.embed_documents([chunk.text for chunk in self.chunks])
            span["output_summary"] = f"{len(self.chunks)} embeddings"
        with self.tracer.span("indexing", input_summary="multimodal index") as span:
            self.vector_store.clear()
            self.vector_store.add(self.chunks, embeddings)
            span["output_summary"] = f"{sum(1 for chunk in self.chunks if chunk.metadata.get('modality') == 'image')} image items"
        return self.chunks

    def retrieve(self, query: str, **kwargs) -> list[RetrievalResult]:
        top_k = kwargs.get("top_k", self.retrieval_config.top_k)
        threshold = kwargs.get("threshold", self.retrieval_config.similarity_threshold or 0.0)
        retrieval_query = build_retrieval_query(query, kwargs.get("history"))
        with self.tracer.span("query_embedding", input_summary=retrieval_query[:80]) as span:
            query_embedding = self.embedder.embed_query(retrieval_query)
            span["output_summary"] = "query vector"
        with self.tracer.span("dense_retrieval", input_summary=f"top_k={top_k}") as span:
            results = self.vector_store.search(query_embedding, k=top_k, threshold=threshold)
            span["output_summary"] = f"{len(results)} mixed-modality hits"
        self.last_extras = {
            "retrieval_query": retrieval_query,
            "by_modality": group_by_modality(results),
            "modalities": [modality_of(result) for result in results],
        }
        return results
