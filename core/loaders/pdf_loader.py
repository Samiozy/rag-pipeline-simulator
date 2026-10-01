from pathlib import Path

from core.models import Document
from .base import DocumentLoader


class PDFLoader(DocumentLoader):
    def load(self, path: Path) -> list[Document]:
        try:
            import fitz
        except ImportError as exc:
            raise ImportError("PDF support requires PyMuPDF. Install it or upload TXT, Markdown, or DOCX.") from exc
        pdf = fitz.open(path)
        docs: list[Document] = []
        for page_idx, page in enumerate(pdf):
            text = page.get_text("text").strip()
            if text:
                docs.append(
                    Document(
                        text=text,
                        source=path.name,
                        metadata={"page": page_idx + 1, "file_type": "pdf", "modality": "text"},
                    )
                )
            for image_idx, _image in enumerate(page.get_images(full=True), start=1):
                caption = f"[Image] {path.name} page {page_idx + 1} figure {image_idx}"
                docs.append(
                    Document(
                        text=caption,
                        source=path.name,
                        metadata={
                            "page": page_idx + 1,
                            "file_type": "pdf-image",
                            "modality": "image",
                            "caption": caption,
                        },
                    )
                )
        return docs
