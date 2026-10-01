import re

from .base import Generator


class ExtractiveGenerator(Generator):
    """Offline fallback. It extracts context sentences relevant to the question."""

    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 700, **kwargs) -> str:
        question_match = re.search(r"QUESTION\n(.+?)\n\nANSWER", prompt, re.S)
        context_match = re.search(r"CONTEXT\n(.+?)\n\nQUESTION", prompt, re.S)
        question = question_match.group(1).strip() if question_match else ""
        context = context_match.group(1).strip() if context_match else prompt
        terms = {word.lower() for word in re.findall(r"[A-Za-z0-9_-]{4,}", question)}
        sentences = re.split(r"(?<=[.!?])\s+", context)
        scored = []
        for sentence in sentences:
            score = sum(term in sentence.lower() for term in terms)
            if score:
                scored.append((score, sentence.strip()))
        selected = [sentence for _, sentence in sorted(scored, reverse=True)[:5]]
        if not selected:
            return "The retrieved context does not contain enough information to answer this question."
        return " ".join(selected)
