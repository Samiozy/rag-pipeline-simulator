from contextlib import contextmanager
from time import perf_counter
from typing import Any, Iterator, Optional

from core.models import TraceEvent


class Tracer:
    """Collects stage timings for the inspector. Strategies should not depend on Streamlit."""

    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def record(
        self,
        stage: str,
        input_summary: str,
        output_summary: str,
        duration_ms: float,
        metadata: Optional[dict[str, Any]] = None,
    ) -> TraceEvent:
        event = TraceEvent(stage, input_summary, output_summary, duration_ms, metadata or {})
        self.events.append(event)
        return event

    @contextmanager
    def span(self, stage: str, input_summary: str = "", metadata: Optional[dict[str, Any]] = None) -> Iterator[dict[str, Any]]:
        payload: dict[str, Any] = {"output_summary": "", "metadata": dict(metadata or {})}
        started = perf_counter()
        try:
            yield payload
        finally:
            duration_ms = (perf_counter() - started) * 1000
            self.record(
                stage,
                input_summary,
                str(payload.get("output_summary") or ""),
                duration_ms,
                payload.get("metadata") or {},
            )

    def clear(self) -> None:
        self.events.clear()
