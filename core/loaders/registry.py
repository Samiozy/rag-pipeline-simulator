from pathlib import Path

from .docx_loader import DOCXLoader
from .image_loader import ImageLoader
from .markdown_loader import MarkdownLoader
from .pdf_loader import PDFLoader
from .text_loader import TextLoader

_LOADERS = {
    ".pdf": PDFLoader,
    ".txt": TextLoader,
    ".md": MarkdownLoader,
    ".markdown": MarkdownLoader,
    ".docx": DOCXLoader,
    ".png": ImageLoader,
    ".jpg": ImageLoader,
    ".jpeg": ImageLoader,
    ".webp": ImageLoader,
    ".gif": ImageLoader,
}


def get_loader(path: Path):
    loader_cls = _LOADERS.get(path.suffix.lower())
    if not loader_cls:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    return loader_cls()
