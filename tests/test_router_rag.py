from core.chunking import RecursiveCharacterChunker
from core.config import RetrievalConfig, RouterConfig
from core.embeddings.hash_embedding import HashEmbedding
from core.generation.extractive import ExtractiveGenerator
from core.models import Document
from core.vectorstores import NumpyVectorStore
from rag_strategies.agentic.router.pipeline import RouterRAG
from rag_strategies.agentic.router.router import HeuristicRouter
from rag_strategies.naive.pipeline import NaiveRAG


def test_router_decides_tool_and_rag():
    router = HeuristicRouter()
    assert router.decide("What is 12 + 5?").route == "tool"
    assert router.decide("What does the document say about chunk overlap?").route == "rag"
    assert router.decide("Hello").route == "direct"


def test_router_rag_exposes_structured_decision():
    docs = [Document("Chunk overlap reduces information loss at chunk boundaries.", "s.txt")]
    naive = NaiveRAG(
        RecursiveCharacterChunker(200, 20),
        HashEmbedding(),
        NumpyVectorStore(),
        ExtractiveGenerator(),
        retrieval=RetrievalConfig(top_k=3, similarity_threshold=-1.0),
    )
    strategy = RouterRAG(naive, ExtractiveGenerator(), retrieval=RouterConfig(top_k=3))
    strategy.ingest(docs)
    output = strategy.run("According to the document, what is chunk overlap used for?")
    decision = output.extras["decision"]
    assert decision.route == "rag"
    assert decision.reason
    assert "chain-of-thought" not in decision.reason.lower()
    calc = strategy.run("What is 3 + 4?")
    assert calc.extras["decision"].route == "tool"
    assert "7" in calc.answer or "calculator" in calc.answer.lower() or calc.context
