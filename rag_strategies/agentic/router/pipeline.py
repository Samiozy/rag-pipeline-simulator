from time import perf_counter
from typing import Optional

from core.config import RouterConfig
from core.context import context_from_results
from core.generation.base import Generator
from core.models import ContextItem, GenerationResult, RetrievalResult
from core.prompts import DEFAULT_SYSTEM_PROMPT, build_rag_prompt
from core.tracing import Tracer
from integrations.memory_adapter import MemoryAdapter
from rag_strategies.base import RAGStrategy
from rag_strategies.agentic.router.router import HeuristicRouter, RouterDecision
from rag_strategies.agentic.router.tools import calculator_context, simulated_web_context
from rag_strategies.naive.pipeline import NaiveRAG


class RouterRAG(RAGStrategy):
    """A decision layer chooses RAG, memory, a tool, or direct generation."""

    key = "agentic_router"
    label = "Agentic RAG — Router"
    description = "A decision layer determines which retrieval path should be used."

    def __init__(
        self,
        document_strategy: NaiveRAG,
        generator: Generator,
        retrieval: Optional[RouterConfig] = None,
        memory: Optional[MemoryAdapter] = None,
        tracer: Optional[Tracer] = None,
    ):
        super().__init__(tracer=tracer)
        self.document_strategy = document_strategy
        self.generator = generator
        self.router = HeuristicRouter()
        self.memory = memory or MemoryAdapter()
        self.retrieval_config = retrieval or RouterConfig()
        self.last_extras: dict = {}
        self.last_decision: Optional[RouterDecision] = None

    def ingest(self, documents):
        chunks = self.document_strategy.ingest(documents)
        self.chunks = chunks
        return chunks

    def retrieve(self, query: str, **kwargs) -> list[RetrievalResult]:
        with self.tracer.span("routing", input_summary=query[:80]) as span:
            decision = self.router.decide(query)
            self.last_decision = decision
            span["output_summary"] = f"route={decision.route}"
            span["metadata"] = {"reason": decision.reason, "confidence": decision.confidence}
        results: list[RetrievalResult] = []
        extra_context: list[ContextItem] = []
        if decision.route == "rag":
            results = self.document_strategy.retrieve(query, top_k=kwargs.get("top_k", self.retrieval_config.top_k), **kwargs)
        elif decision.route == "memory":
            extra_context = self.memory.retrieve(query, k=self.retrieval_config.top_k)
            if not extra_context:
                extra_context = [
                    ContextItem(
                        content="No memory simulator is connected. This route is reserved for a later adapter.",
                        source="memory",
                        source_id="memory_unavailable",
                        metadata={"memory_type": "unavailable"},
                    )
                ]
        elif decision.route == "tool":
            if decision.tool == "calculator":
                extra_context = [calculator_context(query)]
            else:
                extra_context = [simulated_web_context(query)]
        self.last_extras = {
            "retrieval_query": query,
            "decision": decision,
            "available_routes": ["rag", "memory", "tool", "direct"],
            "extra_context": extra_context,
        }
        return results

    def generate(self, query: str, retrieval=None, **kwargs) -> GenerationResult:
        results = retrieval if retrieval is not None else self.retrieve(query, **kwargs)
        context = context_from_results(results)
        extras = kwargs.get("extras") or self.last_extras
        extra_context = list(extras.get("extra_context") or [])
        context = extra_context + context
        system_prompt = kwargs.get("system_prompt") or DEFAULT_SYSTEM_PROMPT
        with self.tracer.span("context_building", input_summary=f"{len(context)} items") as span:
            prompt = build_rag_prompt(query, context, system_prompt=system_prompt, history=kwargs.get("history"))
            span["output_summary"] = f"{len(prompt)} prompt characters"
        started = perf_counter()
        if extras.get("decision") and extras["decision"].route == "direct" and not context:
            answer = "Hello. I can inspect your uploaded documents when you ask a content question."
            generation_ms = 0.0
        else:
            answer = self.generator.generate(prompt, temperature=kwargs.get("temperature", 0.2), max_tokens=kwargs.get("max_tokens", 700))
            generation_ms = (perf_counter() - started) * 1000
        self.tracer.record("generation", f"{len(prompt)} chars", f"{len(answer)} answer chars", generation_ms)
        extras = dict(extras)
        extras["decision"] = extras.get("decision") or self.last_decision
        return GenerationResult(
            question=query,
            answer=answer,
            prompt=prompt,
            retrieved=results,
            context=context,
            retrieval_query=query,
            retrieval_ms=float(kwargs.get("retrieval_ms", 0.0)),
            generation_ms=generation_ms,
            traces=list(self.tracer.events),
            extras=extras,
            strategy=self.key,
        )
