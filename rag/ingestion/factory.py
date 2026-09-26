from pathlib import Path
from .pdf_loader import PDFLoader
from .text_loader import TextLoader
from .docx_loader import DOCXLoader


_LOADERS = {
    ".pdf": PDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
    ".docx": DOCXLoader,
}


def get_loader(path: Path):
    loader_cls = _LOADERS.get(path.suffix.lower())
    if not loader_cls:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    return loader_cls()
