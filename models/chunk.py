from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
