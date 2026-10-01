import os
from typing import Optional

from .base import Generator


class OpenAICompatibleGenerator(Generator):
    """Works with OpenAI and any server exposing an OpenAI-compatible chat-completions API."""

    def __init__(self, model_name: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("The OpenAI package is not installed.") from exc
        self.model_name = model_name
        kwargs = {"api_key": api_key or os.getenv("OPENAI_API_KEY") or "local-not-required"}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 700, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
