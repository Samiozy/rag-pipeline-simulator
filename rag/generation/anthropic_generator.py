import os
from typing import Optional
from .base import Generator


class AnthropicGenerator(Generator):
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        from anthropic import Anthropic
        self.model_name = model_name
        self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 700) -> str:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
