import streamlit as st


def render_sidebar() -> dict:
    st.sidebar.header("Pipeline Configuration")

    chunking = st.sidebar.selectbox("Chunking strategy", ["Recursive Character", "Fixed Character", "Sentence"])
    chunk_size = st.sidebar.slider("Chunk size (characters)", 200, 2400, 800, 100)
    max_overlap = max(0, min(600, chunk_size - 1))
    overlap = st.sidebar.slider("Chunk overlap", 0, max_overlap, min(150, max_overlap), 25, disabled=chunking == "Sentence")

    st.sidebar.divider()
    embedding_provider = st.sidebar.selectbox("Embedding provider", ["Sentence Transformers", "OpenAI"])
    if embedding_provider == "Sentence Transformers":
        embedding_model = st.sidebar.text_input("Embedding model", "sentence-transformers/all-MiniLM-L6-v2")
        embedding_api_key = None
    else:
        embedding_model = st.sidebar.text_input("Embedding model", "text-embedding-3-small")
        embedding_api_key = st.sidebar.text_input("Embedding API key", type="password", help="Optional if OPENAI_API_KEY is set.") or None

    vector_store = st.sidebar.selectbox("Vector store", ["FAISS", "NumPy"])
    top_k = st.sidebar.slider("Top K", 1, 20, 5)
    threshold = st.sidebar.slider("Similarity threshold", -1.0, 1.0, 0.0, 0.05)

    st.sidebar.divider()
    reranker = st.sidebar.selectbox("Reranker", ["None", "Cross Encoder"])
    reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    if reranker == "Cross Encoder":
        reranker_model = st.sidebar.text_input("Reranker model", reranker_model)

    st.sidebar.divider()
    generator_provider = st.sidebar.selectbox(
        "Generator",
        ["Extractive (offline)", "Ollama", "Local LLM (OpenAI-compatible)", "OpenAI / OpenAI-compatible", "Anthropic", "Google Gemini"],
    )
    defaults = {
        "Extractive (offline)": "offline-extractive",
        "Ollama": "llama3.2",
        "Local LLM (OpenAI-compatible)": "llama3.2",
        "OpenAI / OpenAI-compatible": "gpt-4.1-mini",
        "Anthropic": "claude-sonnet-4-5",
        "Google Gemini": "gemini-2.5-flash",
    }
    generator_model = st.sidebar.text_input("Generator model", defaults[generator_provider])
    generator_api_key = None
    base_url = None
    if generator_provider != "Extractive (offline)":
        generator_api_key = st.sidebar.text_input("Generator API key", type="password", help="Optional if the provider environment variable is set.") or None
    if generator_provider in {"Ollama", "Local LLM (OpenAI-compatible)", "OpenAI / OpenAI-compatible"}:
        base_url = st.sidebar.text_input(
            "Base URL (optional)",
            value="http://localhost:11434/v1" if generator_provider in {"Ollama", "Local LLM (OpenAI-compatible)"} else "",
            placeholder="e.g. http://localhost:11434/v1",
        ) or None

    temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.2, 0.1, disabled=generator_provider == "Extractive (offline)")
    max_tokens = st.sidebar.slider("Max output tokens", 100, 3000, 700, 100, disabled=generator_provider == "Extractive (offline)")

    return {
        "chunking": chunking,
        "chunk_size": chunk_size,
        "overlap": overlap if chunking != "Sentence" else 0,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "embedding_api_key": embedding_api_key,
        "vector_store": vector_store,
        "top_k": top_k,
        "threshold": threshold,
        "reranker": reranker,
        "reranker_model": reranker_model,
        "generator_provider": generator_provider,
        "generator_model": generator_model,
        "generator_api_key": generator_api_key,
        "base_url": base_url,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
