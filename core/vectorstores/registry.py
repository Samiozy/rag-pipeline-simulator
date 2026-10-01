from .base import VectorStore
from .numpy_store import NumpyVectorStore


def create_vector_store(name: str) -> VectorStore:
    if name == "NumPy":
        return NumpyVectorStore()
    try:
        from .faiss_store import FAISSVectorStore
    except ImportError as exc:
        raise ImportError("FAISS backend selected but faiss-cpu is not installed. Choose NumPy or install faiss-cpu.") from exc
    return FAISSVectorStore()
