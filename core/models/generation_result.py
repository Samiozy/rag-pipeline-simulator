from dataclasses import dataclass, field
from typing import Any, Optional

from .context_item import ContextItem
from .retrieval_result import RetrievalResult
from .trace import TraceEvent


@dataclass
class GenerationResult:
    """One complete strategy run that the UI can inspect."""

    question: str
    answer: str
    prompt: str
    retrieved: list[RetrievalResult]
    context: list[ContextItem]
    retrieval_query: str = ""
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0
    traces: list[TraceEvent] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)
    strategy: str = ""
    context_chars: int = 0
    context_tokens_est: int = 0

    def __post_init__(self) -> None:
        if not self.total_ms:
            self.total_ms = self.retrieval_ms + self.generation_ms
        if not self.context_chars:
            self.context_chars = sum(len(item.content) for item in self.context)
        if not self.context_tokens_est:
            self.context_tokens_est = max(1, self.context_chars // 4) if self.context_chars else 0
