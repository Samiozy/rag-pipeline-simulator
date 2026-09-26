from dataclasses import dataclass
from time import perf_counter
from typing import Optional
from models import Document, Chunk, RetrievalResult
from rag.chunking.base import Chunker
from rag.embeddings.base import EmbeddingProvider
from rag.vectorstores.base import VectorStore
from rag.generation.base import Generator
from rag.retrieval.reranker import Reranker, NoOpReranker
from rag.prompts import build_rag_prompt


@dataclass
class QueryOutput:
    question: str
    retrieved: list[RetrievalResult]
    prompt: str
    answer: str
    retrieval_ms: float
    generation_ms: float


class RAGPipeline:
    def __init__(
        self,
        chunker: Chunker,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        generator: Generator,
        reranker: Optional[Reranker] = None,
    ):
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.generator = generator
        self.reranker = reranker or NoOpReranker()
        self.chunks: list[Chunk] = []

    def ingest(self, documents: list[Document]) -> list[Chunk]:
        self.chunks = self.chunker.chunk(documents)
        embeddings = self.embedder.embed_documents([chunk.text for chunk in self.chunks])
        self.vector_store.clear()
        self.vector_store.add(self.chunks, embeddings)
        return self.chunks

    def retrieve(self, question: str, top_k: int = 5, threshold: float = 0.0, rerank_top_n: Optional[int] = None) -> list[RetrievalResult]:
        query_embedding = self.embedder.embed_query(question)
        results = self.vector_store.search(query_embedding, k=top_k, threshold=threshold)
        return self.reranker.rerank(question, results, top_n=rerank_top_n)

    def query(
        self,
        question: str,
        top_k: int = 5,
        threshold: float = 0.0,
        rerank_top_n: Optional[int] = None,
        temperature: float = 0.2,
        max_tokens: int = 700,
        system_prompt: Optional[str] = None,
    ) -> QueryOutput:
        started = perf_counter()
        results = self.retrieve(question, top_k, threshold, rerank_top_n)
        retrieval_ms = (perf_counter() - started) * 1000

        context = []
        for result in results:
            metadata = dict(result.chunk.metadata)
            metadata["source"] = result.chunk.source
            context.append((result.chunk.text, metadata))
        prompt = build_rag_prompt(question, context, system_prompt) if system_prompt else build_rag_prompt(question, context)

        started = perf_counter()
        answer = self.generator.generate(prompt, temperature=temperature, max_tokens=max_tokens)
        generation_ms = (perf_counter() - started) * 1000
        return QueryOutput(question, results, prompt, answer, retrieval_ms, generation_ms)
