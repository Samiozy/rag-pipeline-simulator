import numpy as np
from models import Chunk
from rag.vectorstores.numpy_store import NumpyVectorStore


def test_numpy_vector_store_returns_closest_chunk():
    chunks = [Chunk("1", "alpha", "x"), Chunk("2", "beta", "x")]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    store = NumpyVectorStore()
    store.add(chunks, embeddings)
    results = store.search(np.array([1.0, 0.0], dtype="float32"), k=1)
    assert results[0].chunk.id == "1"
    assert results[0].rank == 1
