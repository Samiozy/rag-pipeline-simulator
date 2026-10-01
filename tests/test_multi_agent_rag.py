from core.chunking import RecursiveCharacterChunker
from core.config import MultiAgentConfig, RetrievalConfig
from core.embeddings.hash_embedding import HashEmbedding
from core.generation.extractive import ExtractiveGenerator
from core.models import Document
from core.sparse_retrieval import BM25Retriever
from core.vectorstores import NumpyVectorStore
from rag_strategies.agentic.multi_agent.pipeline import MultiAgentRAG
from rag_strategies.graph.graph_builder import build_graph
from rag_strategies.graph.graph_store import GraphStore
from rag_strategies.naive.pipeline import NaiveRAG


def test_multi_agent_emits_messages_and_answer():
    docs = [Document(
        "Retrieval-Augmented Generation uses embeddings and chunk overlap. "
        "The generator is replaceable and retrieval can be evaluated independently.",
        "s.txt",
    )]
    embedder = HashEmbedding()
    store = NumpyVectorStore()
    sparse = BM25Retriever()
    graph = GraphStore()
    naive = NaiveRAG(
        RecursiveCharacterChunker(180, 20),
        embedder,
        store,
        ExtractiveGenerator(),
        retrieval=RetrievalConfig(top_k=4, similarity_threshold=-1.0),
    )
    naive.ingest(docs)
    build_graph(naive.chunks, store=graph)
    sparse.index(naive.chunks)
    strategy = MultiAgentRAG(
        naive,
        ExtractiveGenerator(),
        sparse_retriever=sparse,
        graph_store=graph,
        retrieval=MultiAgentConfig(top_k=4),
    )
    strategy.chunks = naive.chunks
    output = strategy.run("How does chunk overlap affect retrieval?")
    messages = output.extras["messages"]
    senders = {message.sender for message in messages}
    assert "coordinator" in senders
    assert "document_agent" in senders
    assert "synthesis_agent" in senders
    assert output.answer
    assert output.retrieved
