import numpy as np

from core.vectorstores import NumpyVectorStore


def test_empty_numpy_store_returns_no_results():
    store = NumpyVectorStore()
    assert store.search(np.array([1.0, 0.0], dtype="float32"), k=3) == []
