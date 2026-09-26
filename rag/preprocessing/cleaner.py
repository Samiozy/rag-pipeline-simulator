import re
from models import Document


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_documents(documents: list[Document]) -> list[Document]:
    return [Document(text=clean_text(d.text), source=d.source, metadata=dict(d.metadata)) for d in documents if d.text.strip()]
