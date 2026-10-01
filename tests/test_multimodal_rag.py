from core.chunking import FixedCharacterChunker
from core.config import MultimodalConfig
from core.embeddings.hash_embedding import HashEmbedding
from core.generation.extractive import ExtractiveGenerator
from core.models import Document
from core.vectorstores import NumpyVectorStore
from rag_strategies.multimodal.modality_router import modality_of
from rag_strategies.multimodal.pipeline import MultimodalRAG


def test_multimodal_preserves_image_modality():
    docs = [
        Document("Dense embeddings capture meaning in text.", "note.txt", {"modality": "text", "file_type": "txt"}),
        Document("[Image] diagram.png", "diagram.png", {"modality": "image", "file_type": "png", "caption": "[Image] diagram.png"}),
    ]
    strategy = MultimodalRAG(
        FixedCharacterChunker(200, 20),
        HashEmbedding(),
        NumpyVectorStore(),
        ExtractiveGenerator(),
        retrieval=MultimodalConfig(top_k=5, similarity_threshold=-1.0),
    )
    chunks = strategy.ingest(docs)
    assert any(chunk.metadata.get("modality") == "image" for chunk in chunks)
    results = strategy.retrieve("diagram image")
    assert results
    assert "modalities" in strategy.last_extras
    assert all(modality_of(result) in {"text", "image"} for result in results)
