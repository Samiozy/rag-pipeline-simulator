from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceEvent:
    """Timing and summary for one pipeline stage. Never store hidden chain-of-thought."""

    stage: str
    input_summary: str
    output_summary: str
    duration_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)
