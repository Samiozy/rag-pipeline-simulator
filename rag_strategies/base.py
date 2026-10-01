from abc import ABC, abstractmethod
from time import perf_counter
from typing import Any, Optional

from core.context import context_from_results
from core.models import Chunk, Document, GenerationResult, RetrievalResult
from core.prompts import DEFAULT_SYSTEM_PROMPT, build_rag_prompt
from core.tracing import Tracer


class RAGStrategy(ABC):
    """Common contract for every RAG architecture in the laboratory."""

    key: str = "strategy"
    label: str = "RAG Strategy"
    description: str = ""

    def __init__(self, tracer: Optional[Tracer] = None):
        self.tracer = tracer or Tracer()
        self.chunks: list[Chunk] = []

    @abstractmethod
    def ingest(self, documents: list[Document]) -> list[Chunk]:
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, query: str, **kwargs) -> list[RetrievalResult]:
        raise NotImplementedError

    def generate(self, query: str, retrieval: Optional[list[RetrievalResult]] = None, **kwargs) -> GenerationResult:
        results = retrieval if retrieval is not None else self.retrieve(query, **kwargs)
        context = context_from_results(results)
        system_prompt = kwargs.get("system_prompt") or DEFAULT_SYSTEM_PROMPT
        history = kwargs.get("history")
        with self.tracer.span("prompt_building", input_summary=f"{len(context)} context items") as span:
            prompt = build_rag_prompt(query, context, system_prompt=system_prompt, history=history)
            span["output_summary"] = f"{len(prompt)} prompt characters"
        started = perf_counter()
        answer = self.generator.generate(  # type: ignore[attr-defined]
            prompt,
            temperature=kwargs.get("temperature", 0.2),
            max_tokens=kwargs.get("max_tokens", 700),
        )
        generation_ms = (perf_counter() - started) * 1000
        self.tracer.record("generation", f"{len(prompt)} chars", f"{len(answer)} answer chars", generation_ms)
        extras = kwargs.get("extras") or {}
        return GenerationResult(
            question=query,
            answer=answer,
            prompt=prompt,
            retrieved=results,
            context=context,
            retrieval_query=kwargs.get("retrieval_query", query),
            retrieval_ms=float(kwargs.get("retrieval_ms", 0.0)),
            generation_ms=generation_ms,
            traces=list(self.tracer.events),
            extras=extras,
            strategy=self.key,
        )

    def run(self, query: str, **kwargs) -> GenerationResult:
        self.tracer.clear()
        started = perf_counter()
        retrieval = self.retrieve(query, **kwargs)
        retrieval_ms = (perf_counter() - started) * 1000
        extras = getattr(self, "last_extras", {})
        result = self.generate(
            query,
            retrieval=retrieval,
            retrieval_ms=retrieval_ms,
            extras=extras,
            retrieval_query=extras.get("retrieval_query", query),
            **kwargs,
        )
        result.retrieval_ms = retrieval_ms
        result.total_ms = retrieval_ms + result.generation_ms
        result.retrieval_query = extras.get("retrieval_query", query)
        return result
