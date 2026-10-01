import streamlit as st

from core.config import GraphConfig, HybridConfig, MultiAgentConfig, MultimodalConfig, RerankConfig, RetrievalConfig, RouterConfig
from rag_strategies import available_strategies, strategy_labels

RUN_MODES = {
    "offline": "Offline — no API key",
    "local": "Local LLM — Ollama",
    "cloud": "Cloud / custom",
}

GENERATOR_DEFAULTS = {
    "Extractive (offline)": "offline-extractive",
    "Ollama": "llama3.2",
    "Local LLM (OpenAI-compatible)": "llama3.2",
    "OpenAI / OpenAI-compatible": "gpt-4.1-mini",
    "Anthropic": "claude-sonnet-4-5",
    "Google Gemini": "gemini-2.5-flash",
}

CLOUD_GENERATORS = [
    "OpenAI / OpenAI-compatible",
    "Anthropic",
    "Google Gemini",
    "Ollama",
    "Local LLM (OpenAI-compatible)",
    "Extractive (offline)",
]

PAGES = [
    ("overview", "Overview"),
    ("naive", "Naive RAG"),
    ("rerank", "Retrieve + Rerank"),
    ("hybrid", "Hybrid RAG"),
    ("graph", "Graph RAG"),
    ("multimodal", "Multimodal RAG"),
    ("agentic_router", "Agentic RAG"),
    ("multi_agent", "Multi-Agent RAG"),
    ("compare", "Compare"),
]


def _apply_preset_defaults(run_mode: str) -> None:
    previous = st.session_state.get("applied_run_mode")
    if previous == run_mode:
        return
    st.session_state["applied_run_mode"] = run_mode
    if run_mode == "offline":
        st.session_state["embedding_provider"] = "Sentence Transformers"
        st.session_state["generator_provider"] = "Extractive (offline)"
    elif run_mode == "local":
        st.session_state["embedding_provider"] = "Sentence Transformers"
        st.session_state["generator_provider"] = "Ollama"
        st.session_state["generator_model"] = GENERATOR_DEFAULTS["Ollama"]
        st.session_state["base_url"] = "http://localhost:11434/v1"
    else:
        st.session_state["embedding_provider"] = "Sentence Transformers"
        st.session_state["generator_provider"] = "OpenAI / OpenAI-compatible"
        st.session_state["generator_model_cloud"] = GENERATOR_DEFAULTS["OpenAI / OpenAI-compatible"]


def render_navigation() -> str:
    labels = {key: label for key, label in PAGES}
    page = st.sidebar.radio(
        "Laboratory",
        options=[key for key, _ in PAGES],
        format_func=lambda key: labels[key],
        key="lab_page",
    )
    return page


def render_sidebar(page: str) -> dict:
    st.sidebar.markdown("### How should answers be generated?")
    run_mode = st.sidebar.radio(
        "Run mode",
        options=list(RUN_MODES.keys()),
        format_func=lambda key: RUN_MODES[key],
        key="run_mode_choice",
        label_visibility="collapsed",
    )
    _apply_preset_defaults(run_mode)

    with st.sidebar.expander("Shared: chunking", expanded=True):
        chunking = st.selectbox("How to split text", ["Recursive Character", "Fixed Character", "Sentence"])
        chunk_size = st.slider("Target chunk size (characters)", 200, 2400, 800, 100)
        max_overlap = max(0, min(600, chunk_size - 1))
        overlap = st.slider("Overlap between chunks", 0, max_overlap, min(150, max_overlap), 25, disabled=chunking == "Sentence")

    with st.sidebar.expander("Shared: embeddings and index", expanded=False):
        embedding_provider = st.selectbox("Embedding model source", ["Sentence Transformers", "OpenAI"], key="embedding_provider")
        if embedding_provider == "Sentence Transformers":
            embedding_model = st.text_input("Embedding model", "sentence-transformers/all-MiniLM-L6-v2")
            embedding_api_key = None
        else:
            embedding_model = st.text_input("Embedding model", "text-embedding-3-small")
            embedding_api_key = st.text_input("Embedding API key", type="password") or None
        vector_store = st.selectbox("Where to store vectors", ["FAISS", "NumPy"])

    generator_provider, generator_model, generator_api_key, base_url, temperature, max_tokens = _render_generator_settings(run_mode)

    retrieval = RetrievalConfig()
    rerank = RerankConfig()
    hybrid = HybridConfig()
    graph = GraphConfig()
    multimodal = MultimodalConfig()
    router = RouterConfig()
    multi_agent = MultiAgentConfig()
    reranker_name = "None"
    reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    if page in {"naive", "compare"}:
        with st.sidebar.expander("Naive retrieval", expanded=page == "naive"):
            retrieval.top_k = st.slider("Top-K", 1, 20, 5, key="naive_top_k")
            retrieval.similarity_threshold = st.slider("Minimum similarity", -1.0, 1.0, 0.0, 0.05, key="naive_threshold")

    if page in {"rerank", "compare"}:
        with st.sidebar.expander("Rerank retrieval", expanded=page == "rerank"):
            rerank.candidate_k = st.slider("Candidate K (first pass)", 5, 40, 20, key="rerank_candidate_k")
            rerank.top_k = st.slider("Final Top-K", 1, 20, 5, key="rerank_top_k")
            rerank.similarity_threshold = st.slider("Minimum similarity", -1.0, 1.0, 0.0, 0.05, key="rerank_threshold")
            reranker_name = st.selectbox("Reranker", ["None", "Cross Encoder", "Keyword (offline)"], key="reranker_name")
            if reranker_name == "Cross Encoder":
                reranker_model = st.text_input("Reranker model", reranker_model)

    if page in {"hybrid", "compare"}:
        with st.sidebar.expander("Hybrid retrieval", expanded=page == "hybrid"):
            hybrid.dense_k = st.slider("Dense Top-K", 1, 30, 10, key="hybrid_dense_k")
            hybrid.sparse_k = st.slider("Sparse / BM25 Top-K", 1, 30, 10, key="hybrid_sparse_k")
            hybrid.final_k = st.slider("Final Top-K", 1, 20, 5, key="hybrid_final_k")
            hybrid.fusion_method = st.selectbox("Fusion method", ["rrf", "weighted", "normalized"], format_func=lambda item: {
                "rrf": "Reciprocal Rank Fusion",
                "weighted": "Weighted score",
                "normalized": "Normalized score",
            }[item], key="fusion_method")
            hybrid.dense_weight = st.slider("Dense weight", 0.0, 1.0, 0.6, 0.05, key="dense_weight")
            hybrid.sparse_weight = 1.0 - hybrid.dense_weight
            st.caption(f"Sparse weight is {hybrid.sparse_weight:.2f} (1 − dense weight).")

    if page in {"graph", "compare", "multi_agent"}:
        with st.sidebar.expander("Graph retrieval", expanded=page == "graph"):
            graph.top_k = st.slider("Graph Top-K", 1, 20, 5, key="graph_top_k")
            graph.traversal_depth = st.slider("Traversal depth", 1, 4, 2, key="graph_depth")

    if page in {"multimodal", "compare"}:
        with st.sidebar.expander("Multimodal retrieval", expanded=page == "multimodal"):
            multimodal.top_k = st.slider("Multimodal Top-K", 1, 20, 5, key="mm_top_k")
            multimodal.similarity_threshold = st.slider("Minimum similarity", -1.0, 1.0, 0.0, 0.05, key="mm_threshold")
            st.caption("Images are stored as captioned items with modality=image. CLIP is not required.")

    if page in {"agentic_router", "compare"}:
        with st.sidebar.expander("Router", expanded=page == "agentic_router"):
            router.top_k = st.slider("RAG Top-K when routed to documents", 1, 20, 5, key="router_top_k")
            st.caption("Routes: rag, memory, tool (calculator / simulated web), direct. Decision summaries are shown, not hidden chain-of-thought.")

    if page in {"multi_agent", "compare"}:
        with st.sidebar.expander("Multi-agent", expanded=page == "multi_agent"):
            multi_agent.top_k = st.slider("Merged Top-K", 1, 20, 5, key="ma_top_k")
            multi_agent.enable_search_agent = st.checkbox("Lexical search agent", value=True, key="ma_search")
            multi_agent.enable_graph_agent = st.checkbox("Graph agent", value=True, key="ma_graph")

    compare_a, compare_b = "naive", "hybrid"
    if page == "compare":
        labels = strategy_labels()
        keys = available_strategies()
        compare_a = st.sidebar.selectbox("Pipeline A", keys, format_func=lambda key: labels[key], key="compare_a")
        compare_b = st.sidebar.selectbox("Pipeline B", keys, index=min(2, len(keys) - 1), format_func=lambda key: labels[key], key="compare_b")

    return {
        "run_mode": run_mode,
        "page": page,
        "chunking": chunking,
        "chunk_size": chunk_size,
        "overlap": overlap if chunking != "Sentence" else 0,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "embedding_api_key": embedding_api_key,
        "vector_store": vector_store,
        "generator_provider": generator_provider,
        "generator_model": generator_model,
        "generator_api_key": generator_api_key,
        "base_url": base_url,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reranker_name": reranker_name,
        "reranker_model": reranker_model,
        "retrieval": retrieval,
        "rerank": rerank,
        "hybrid": hybrid,
        "graph": graph,
        "multimodal": multimodal,
        "router": router,
        "multi_agent": multi_agent,
        "compare_a": compare_a,
        "compare_b": compare_b,
    }


def _render_generator_settings(run_mode: str):
    with st.sidebar.expander("Shared: generator", expanded=True):
        if run_mode == "offline":
            st.markdown("**Extractive (offline)**")
            st.caption("No model call. The app quotes and stitches the best retrieved passages.")
            return "Extractive (offline)", "offline-extractive", None, None, 0.2, 700

        if run_mode == "local":
            generator_provider = st.selectbox("Local server", ["Ollama", "Local LLM (OpenAI-compatible)"], key="generator_provider_local")
            generator_model = st.text_input("Model name", GENERATOR_DEFAULTS[generator_provider], key="generator_model")
            base_url = st.text_input("Server URL", value=st.session_state.get("base_url", "http://localhost:11434/v1"), key="base_url") or None
            generator_api_key = st.text_input("API key (usually blank)", type="password") or None
        else:
            generator_provider = st.selectbox("Provider", CLOUD_GENERATORS, key="generator_provider")
            generator_model = st.text_input("Model name", GENERATOR_DEFAULTS[generator_provider], key="generator_model_cloud")
            generator_api_key = None
            base_url = None
            if generator_provider != "Extractive (offline)":
                generator_api_key = st.text_input("API key", type="password") or None
            if generator_provider in {"Ollama", "Local LLM (OpenAI-compatible)", "OpenAI / OpenAI-compatible"}:
                default_url = "http://localhost:11434/v1" if generator_provider in {"Ollama", "Local LLM (OpenAI-compatible)"} else ""
                base_url = st.text_input("Base URL (optional)", value=default_url) or None

        extractive = generator_provider == "Extractive (offline)"
        temperature = st.slider("Creativity (temperature)", 0.0, 1.5, 0.2, 0.1, disabled=extractive)
        max_tokens = st.slider("Maximum answer length (tokens)", 100, 3000, 700, 100, disabled=extractive)
        return generator_provider, generator_model, generator_api_key, base_url, temperature, max_tokens
