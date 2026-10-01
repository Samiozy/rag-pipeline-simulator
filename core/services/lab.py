from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from core.chunking import make_chunker
from core.config import (
    GraphConfig,
    HybridConfig,
    MultiAgentConfig,
    MultimodalConfig,
    RerankConfig,
    RetrievalConfig,
    RouterConfig,
)
from core.embeddings import create_embedder
from core.embeddings.base import EmbeddingProvider
from core.generation import create_generator
from core.generation.base import Generator
from core.loaders import get_loader
from core.models import Chunk, Document
from core.preprocessing import clean_documents
from core.reranking import CrossEncoderReranker, IdentityReranker, KeywordReranker, Reranker
from core.sparse_retrieval import BM25Retriever
from core.tracing import Tracer
from core.vectorstores import create_vector_store
from rag_strategies.agentic.multi_agent.pipeline import MultiAgentRAG
from rag_strategies.agentic.router.pipeline import RouterRAG
from rag_strategies.base import RAGStrategy
from rag_strategies.graph.graph_builder import build_graph
from rag_strategies.graph.graph_store import GraphStore
from rag_strategies.graph.pipeline import GraphRAG
from rag_strategies.hybrid.pipeline import HybridRAG
from rag_strategies.multimodal.pipeline import MultimodalRAG
from rag_strategies.naive.pipeline import NaiveRAG
from rag_strategies.rerank.pipeline import RerankRAG


@dataclass
class LabComponents:
    chunker: object
    embedder: EmbeddingProvider
    vector_store: object
    generator: Generator
    reranker: Reranker
    sparse_retriever: BM25Retriever
    graph_store: GraphStore
    tracer: Tracer


def load_documents(paths: list[Path], display_names: Optional[list[str]] = None) -> list[Document]:
    documents: list[Document] = []
    names = display_names or [path.name for path in paths]
    for path, name in zip(paths, names):
        loader = get_loader(path)
        loaded = loader.load(path)
        for document in loaded:
            document.source = name
        documents.extend(loaded)
    if not documents:
        raise ValueError("No text could be extracted from the uploaded files.")
    return clean_documents(documents)


def build_components(
    chunking: str,
    chunk_size: int,
    overlap: int,
    embedding_provider: str,
    embedding_model: str,
    embedding_api_key: Optional[str],
    vector_store: str,
    generator_provider: str,
    generator_model: str,
    generator_api_key: Optional[str],
    base_url: Optional[str],
    reranker_name: str = "None",
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> LabComponents:
    reranker: Reranker
    if reranker_name == "Cross Encoder":
        reranker = CrossEncoderReranker(reranker_model)
    elif reranker_name == "Keyword (offline)":
        reranker = KeywordReranker()
    else:
        reranker = IdentityReranker()
    return LabComponents(
        chunker=make_chunker(chunking, chunk_size, overlap),
        embedder=create_embedder(embedding_provider, embedding_model, embedding_api_key),
        vector_store=create_vector_store(vector_store),
        generator=create_generator(generator_provider, generator_model, generator_api_key, base_url),
        reranker=reranker,
        sparse_retriever=BM25Retriever(),
        graph_store=GraphStore(),
        tracer=Tracer(),
    )


def populate_index(components: LabComponents, documents: list[Document]) -> list[Chunk]:
    """Chunk, embed, and index once so multiple strategies can share the same corpus."""
    for document in documents:
        document.metadata.setdefault("modality", "text")
    with components.tracer.span("chunking", input_summary=f"{len(documents)} documents") as span:
        chunks = components.chunker.chunk(documents)
        for chunk in chunks:
            chunk.metadata.setdefault("modality", document_modality(chunk))
        span["output_summary"] = f"{len(chunks)} chunks"
    with components.tracer.span("embedding", input_summary=f"{len(chunks)} chunks") as span:
        embeddings = components.embedder.embed_documents([chunk.text for chunk in chunks])
        span["output_summary"] = f"{len(chunks)} embeddings"
        span["metadata"] = {"shape": list(embeddings.shape)}
    with components.tracer.span("indexing", input_summary="dense + BM25 + graph") as span:
        components.vector_store.clear()
        components.vector_store.add(chunks, embeddings)
        components.sparse_retriever.index(chunks)
        build_graph(chunks, store=components.graph_store)
        span["output_summary"] = f"{len(chunks)} vectors, BM25 docs, {len(components.graph_store.nodes)} graph nodes"
    return chunks


def document_modality(chunk: Chunk) -> str:
    return str(chunk.metadata.get("modality") or "text")


def _naive(components: LabComponents, retrieval: Optional[RetrievalConfig] = None) -> NaiveRAG:
    return NaiveRAG(
        components.chunker,
        components.embedder,
        components.vector_store,
        components.generator,
        retrieval=retrieval or RetrievalConfig(),
        tracer=components.tracer,
    )


def build_strategy(
    key: str,
    components: LabComponents,
    retrieval: Optional[RetrievalConfig] = None,
    rerank: Optional[RerankConfig] = None,
    hybrid: Optional[HybridConfig] = None,
    graph: Optional[GraphConfig] = None,
    multimodal: Optional[MultimodalConfig] = None,
    router: Optional[RouterConfig] = None,
    multi_agent: Optional[MultiAgentConfig] = None,
) -> RAGStrategy:
    if key == "rerank":
        return RerankRAG(
            components.chunker,
            components.embedder,
            components.vector_store,
            components.generator,
            reranker=components.reranker,
            retrieval=rerank or RerankConfig(),
            tracer=components.tracer,
        )
    if key == "hybrid":
        return HybridRAG(
            components.chunker,
            components.embedder,
            components.vector_store,
            components.sparse_retriever,
            components.generator,
            retrieval=hybrid or HybridConfig(),
            tracer=components.tracer,
        )
    if key == "graph":
        strategy = GraphRAG(
            components.chunker,
            components.generator,
            graph_store=components.graph_store,
            retrieval=graph or GraphConfig(),
            tracer=components.tracer,
        )
        return strategy
    if key == "multimodal":
        return MultimodalRAG(
            components.chunker,
            components.embedder,
            components.vector_store,
            components.generator,
            retrieval=multimodal or MultimodalConfig(),
            tracer=components.tracer,
        )
    if key == "agentic_router":
        return RouterRAG(
            _naive(components, retrieval),
            components.generator,
            retrieval=router or RouterConfig(),
            tracer=components.tracer,
        )
    if key == "multi_agent":
        return MultiAgentRAG(
            _naive(components, retrieval),
            components.generator,
            sparse_retriever=components.sparse_retriever,
            graph_store=components.graph_store,
            retrieval=multi_agent or MultiAgentConfig(),
            tracer=components.tracer,
        )
    return _naive(components, retrieval)
