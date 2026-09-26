from typing import Optional
from .base import EmbeddingProvider


def create_embedder(provider: str, model_name: str, api_key: Optional[str] = None):
    from .factory import create_embedder as _create_embedder
    return _create_embedder(provider, model_name, api_key)


__all__ = ["EmbeddingProvider", "create_embedder"]
