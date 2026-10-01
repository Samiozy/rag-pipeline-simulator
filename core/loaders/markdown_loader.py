from .text_loader import TextLoader


class MarkdownLoader(TextLoader):
    """Markdown is ingested as UTF-8 text. Kept as its own loader so the registry stays explicit."""
