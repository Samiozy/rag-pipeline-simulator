from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A cleaned ingestion unit, typically a file or a PDF page."""

    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
