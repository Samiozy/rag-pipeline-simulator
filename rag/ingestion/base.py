from abc import ABC, abstractmethod
from pathlib import Path
from models import Document


class DocumentLoader(ABC):
    @abstractmethod
    def load(self, path: Path) -> list[Document]:
        raise NotImplementedError
