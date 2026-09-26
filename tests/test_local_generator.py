from rag.generation.factory import create_generator
from rag.generation.openai_compatible import OpenAICompatibleGenerator


def test_create_local_generator_uses_openai_compatible_backend():
    generator = create_generator(
        "Local LLM (OpenAI-compatible)",
        model_name="llama3.2",
        base_url="http://localhost:11434/v1",
    )

    assert isinstance(generator, OpenAICompatibleGenerator)


def test_create_ollama_generator_uses_local_default_url():
    generator = create_generator("Ollama", model_name="llama3.2")

    assert isinstance(generator, OpenAICompatibleGenerator)
