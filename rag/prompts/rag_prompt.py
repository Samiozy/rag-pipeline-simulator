DEFAULT_SYSTEM_PROMPT = """You are a retrieval-augmented assistant. Answer using only the supplied context. If the context does not contain enough information, say that the answer is not supported by the retrieved context. Cite sources using [Source N]."""


def build_rag_prompt(question: str, retrieved_context: list[tuple[str, dict]], system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    blocks = []
    for i, (text, metadata) in enumerate(retrieved_context, 1):
        source = metadata.get("source", "unknown")
        page = metadata.get("page")
        label = f"{source}, page {page}" if page else source
        blocks.append(f"[Source {i}: {label}]\n{text}")
    context = "\n\n".join(blocks) if blocks else "No context retrieved."
    return f"{system_prompt}\n\nCONTEXT\n{context}\n\nQUESTION\n{question}\n\nANSWER"
