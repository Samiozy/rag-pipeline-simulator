from typing import Iterable, Sequence, Union

from core.models import ContextItem, RetrievalResult

ContextInput = Union[ContextItem, RetrievalResult, tuple]


def context_from_results(results: Sequence[RetrievalResult], source: str = "rag") -> list[ContextItem]:
    items: list[ContextItem] = []
    for result in results:
        page = result.chunk.metadata.get("page")
        source_id = f"{result.chunk.source}:{result.chunk.id}"
        metadata = dict(result.chunk.metadata)
        metadata.update({
            "source": result.chunk.source,
            "chunk_id": result.chunk.id,
            "rank": result.rank,
            "channel": result.channel,
        })
        if page is not None:
            source_id = f"{result.chunk.source}:p{page}:{result.chunk.id}"
        items.append(
            ContextItem(
                content=result.chunk.text,
                source=source,
                source_id=source_id,
                score=result.score,
                metadata=metadata,
            )
        )
    return items


def normalize_context(retrieved_context: Iterable[ContextInput]) -> list[tuple[str, dict]]:
    """Accept ContextItems, RetrievalResults, or (text, metadata) tuples."""
    blocks: list[tuple[str, dict]] = []
    for item in retrieved_context:
        if isinstance(item, ContextItem):
            metadata = dict(item.metadata)
            metadata.setdefault("source", metadata.get("source") or item.source)
            blocks.append((item.content, metadata))
        elif isinstance(item, RetrievalResult):
            metadata = dict(item.chunk.metadata)
            metadata["source"] = item.chunk.source
            blocks.append((item.chunk.text, metadata))
        else:
            text, metadata = item
            blocks.append((text, dict(metadata)))
    return blocks
