from dataclasses import dataclass, field
from typing import Any

from .generation_result import GenerationResult


@dataclass
class ExperimentResult:
    """Side-by-side comparison of two independent strategy runs."""

    query: str
    pipeline_a: GenerationResult
    pipeline_b: GenerationResult
    overlap: dict[str, Any] = field(default_factory=dict)
    config_a: dict[str, Any] = field(default_factory=dict)
    config_b: dict[str, Any] = field(default_factory=dict)
