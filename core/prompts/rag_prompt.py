from typing import Iterable, Optional

from core.context import ContextInput, normalize_context

DEFAULT_SYSTEM_PROMPT = """You are a retrieval-augmented assistant. Answer using only the supplied context. If the user asks a follow-up, resolve pronouns and references from the conversation so far, but still answer only from the retrieved context. If the context does not contain enough information, say that the answer is not supported by the retrieved context. Cite sources using [Source N]."""

HISTORY_TURN_LIMIT = 6
HISTORY_ANSWER_CHARS = 600
RETRIEVAL_HISTORY_TURNS = 3


def build_retrieval_query(
    question: str,
    history: Optional[list[tuple[str, str]]] = None,
    max_turns: int = RETRIEVAL_HISTORY_TURNS,
) -> str:
    """Expand a follow-up question with prior questions so retrieval is not pronoun-blind."""
    if not history:
        return question
    prior = [item[0].strip() for item in history[-max_turns:] if item and item[0].strip()]
    if not prior:
        return question
    previous = "\n".join(f"- {item}" for item in prior)
    return f"{question}\n\nPrevious questions:\n{previous}"


def _conversation_block(history: list[tuple[str, str]], max_turns: int, answer_chars: int) -> str:
    lines = []
    for question, answer in history[-max_turns:]:
        clipped = (answer or "").strip()
        if len(clipped) > answer_chars:
            clipped = clipped[:answer_chars].rstrip() + "…"
        lines.append(f"User: {question.strip()}")
        lines.append(f"Assistant: {clipped or '(no answer)'}")
    return "CONVERSATION\n" + "\n".join(lines)


def build_rag_prompt(
    question: str,
    retrieved_context: Iterable[ContextInput],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    history: Optional[list[tuple[str, str]]] = None,
) -> str:
    blocks = []
    for index, (text, metadata) in enumerate(normalize_context(retrieved_context), 1):
        source = metadata.get("source", "unknown")
        page = metadata.get("page")
        label = f"{source}, page {page}" if page else source
        blocks.append(f"[Source {index}: {label}]\n{text}")
    context = "\n\n".join(blocks) if blocks else "No context retrieved."
    sections = [system_prompt]
    if history:
        sections.append(_conversation_block(history, HISTORY_TURN_LIMIT, HISTORY_ANSWER_CHARS))
    sections.append(f"CONTEXT\n{context}")
    sections.append(f"QUESTION\n{question}\n\nANSWER")
    return "\n\n".join(sections)
