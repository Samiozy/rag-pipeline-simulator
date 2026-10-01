from dataclasses import dataclass, field
from typing import Any, Optional

from .chunk import Chunk


@dataclass
class RetrievalResult:
    """A ranked chunk from one retrieval channel."""

    chunk: Chunk
    score: float
    rank: int = 0
    channel: str = "dense"
    metadata: dict[str, Any] = field(default_factory=dict)
    original_rank: Optional[int] = None
