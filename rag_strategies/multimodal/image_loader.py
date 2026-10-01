from pathlib import Path

from core.models import Document


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


def is_image_path(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def load_image_document(path: Path) -> Document:
    """Caption-only image document. Pixel models are optional and not required."""
    caption = f"[Image] {path.name}"
    return Document(
        text=caption,
        source=path.name,
        metadata={"modality": "image", "file_type": path.suffix.lstrip("."), "caption": caption},
    )


def extract_pdf_image_documents(path: Path) -> list[Document]:
    """Record PDF-embedded images as caption documents so modality is inspectable without CLIP."""
    try:
        import fitz
    except ImportError:
        return []
    documents: list[Document] = []
    pdf = fitz.open(path)
    for page_idx, page in enumerate(pdf):
        images = page.get_images(full=True)
        for image_idx, _image in enumerate(images, start=1):
            caption = f"[Image] {path.name} page {page_idx + 1} figure {image_idx}"
            documents.append(
                Document(
                    text=caption,
                    source=path.name,
                    metadata={
                        "modality": "image",
                        "file_type": "pdf-image",
                        "page": page_idx + 1,
                        "caption": caption,
                    },
                )
            )
    return documents
