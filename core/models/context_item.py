from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ContextItem:
    """Provider-neutral context used by prompts and later memory adapters."""

    content: str
    source: str
    source_id: str
    score: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)
