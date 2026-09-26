from typing import Optional
from .base import EmbeddingProvider


def create_embedder(provider: str, model_name: str, api_key: Optional[str] = None) -> EmbeddingProvider:
    if provider == "OpenAI":
        from .openai_embedding import OpenAIEmbedding
        return OpenAIEmbedding(model_name=model_name, api_key=api_key)
    from .sentence_transformer import SentenceTransformerEmbedding
    return SentenceTransformerEmbedding(model_name=model_name)
