from dataclasses import dataclass
import os
from typing import Optional


@dataclass(frozen=True)
class AppSettings:
    app_title: str = "RAG Pipeline Simulator"
    default_chunk_size: int = 800
    default_chunk_overlap: int = 150
    default_top_k: int = 5
    default_threshold: float = 0.0
    default_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    default_generator: str = "extractive"

    @property
    def openai_api_key(self) -> Optional[str]:
        return os.getenv("OPENAI_API_KEY")

    @property
    def anthropic_api_key(self) -> Optional[str]:
        return os.getenv("ANTHROPIC_API_KEY")

    @property
    def google_api_key(self) -> Optional[str]:
        return os.getenv("GOOGLE_API_KEY")


settings = AppSettings()
