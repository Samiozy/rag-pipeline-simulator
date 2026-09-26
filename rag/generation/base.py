from abc import ABC, abstractmethod


class Generator(ABC):
    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 700) -> str:
        raise NotImplementedError
