from pathlib import Path

from core.models import Document
from .base import DocumentLoader


class TextLoader(DocumentLoader):
    def load(self, path: Path) -> list[Document]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [Document(text=text, source=path.name, metadata={"file_type": path.suffix.lstrip("."), "modality": "text"})]
