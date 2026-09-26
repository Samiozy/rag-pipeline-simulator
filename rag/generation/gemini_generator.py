import os
from typing import Optional
from .base import Generator


class GeminiGenerator(Generator):
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        from google import genai
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key or os.getenv("GOOGLE_API_KEY"))

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 700) -> str:
        from google.genai import types
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=temperature, max_output_tokens=max_tokens),
        )
        return response.text or ""
