from core.models import RetrievalResult


def modality_of(result: RetrievalResult) -> str:
    return str(result.chunk.metadata.get("modality") or "text")


def group_by_modality(results: list[RetrievalResult]) -> dict[str, list[RetrievalResult]]:
    grouped: dict[str, list[RetrievalResult]] = {}
    for result in results:
        grouped.setdefault(modality_of(result), []).append(result)
    return grouped
