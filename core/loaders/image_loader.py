from pathlib import Path

from core.models import Document
from .base import DocumentLoader


class ImageLoader(DocumentLoader):
    """Ingests an image as a caption document. Pixel encoders are optional later."""

    def load(self, path: Path) -> list[Document]:
        caption = f"[Image] {path.name}"
        return [
            Document(
                text=caption,
                source=path.name,
                metadata={"modality": "image", "file_type": path.suffix.lstrip("."), "caption": caption},
            )
        ]
