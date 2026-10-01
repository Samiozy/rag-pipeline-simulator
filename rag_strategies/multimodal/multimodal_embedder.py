"""Text-caption embeddings for images. Optional CLIP can be added later without changing strategies."""

from core.embeddings.base import EmbeddingProvider


class CaptionEmbeddingAdapter:
    """Uses a text embedder on image captions so multimodal mode works offline."""

    def __init__(self, embedder: EmbeddingProvider):
        self.embedder = embedder

    def embed_documents(self, texts: list[str]):
        return self.embedder.embed_documents(texts)

    def embed_query(self, text: str):
        return self.embedder.embed_query(text)
