from pathlib import Path
import fitz
from models import Document
from .base import DocumentLoader


class PDFLoader(DocumentLoader):
    def load(self, path: Path) -> list[Document]:
        pdf = fitz.open(path)
        docs: list[Document] = []
        for page_idx, page in enumerate(pdf):
            text = page.get_text("text").strip()
            if text:
                docs.append(
                    Document(
                        text=text,
                        source=path.name,
                        metadata={"page": page_idx + 1, "file_type": "pdf"},
                    )
                )
        return docs
