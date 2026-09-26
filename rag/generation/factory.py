from typing import Optional
from .base import Generator
from .extractive import ExtractiveGenerator


def create_generator(
    provider: str,
    model_name: str = "",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Generator:
    if provider in {"OpenAI / OpenAI-compatible", "Local LLM (OpenAI-compatible)", "Ollama"}:
        from .openai_compatible import OpenAICompatibleGenerator
        resolved_base_url = base_url or (
            "http://localhost:11434/v1" if provider in {"Local LLM (OpenAI-compatible)", "Ollama"} else None
        )
        return OpenAICompatibleGenerator(model_name=model_name, api_key=api_key, base_url=resolved_base_url)
    if provider == "Anthropic":
        from .anthropic_generator import AnthropicGenerator
        return AnthropicGenerator(model_name=model_name, api_key=api_key)
    if provider == "Google Gemini":
        from .gemini_generator import GeminiGenerator
        return GeminiGenerator(model_name=model_name, api_key=api_key)
    return ExtractiveGenerator()
