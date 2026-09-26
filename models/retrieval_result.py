from dataclasses import dataclass
from .chunk import Chunk


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    rank: int = 0
