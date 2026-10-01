from core.embeddings.hash_embedding import HashEmbedding


def test_hash_embedding_shape_and_query():
    embedder = HashEmbedding(dim=16)
    docs = embedder.embed_documents(["alpha", "beta"])
    query = embedder.embed_query("alpha")
    assert docs.shape == (2, 16)
    assert query.shape == (16,)
