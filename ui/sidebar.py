import streamlit as st

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


def render_sidebar() -> dict:
    st.sidebar.markdown("### How should answers be generated?")
    run_mode = st.sidebar.radio(
        "Run mode",
        options=list(RUN_MODES.keys()),
        format_func=lambda key: RUN_MODES[key],
        key="run_mode_choice",
        label_visibility="collapsed",
        help="Start offline to inspect retrieval. Switch to a local or cloud model when you want generated prose.",
    )
    _apply_preset_defaults(run_mode)

    if run_mode == "offline":
        st.sidebar.caption("Answers are assembled from the retrieved passages. Use this to judge retrieval quality first.")
    elif run_mode == "local":
        st.sidebar.caption("Uses a model on your machine. Start Ollama (or another OpenAI-compatible server) first.")
    else:
        st.sidebar.caption("Uses a hosted provider. Paste an API key here or set it in your environment.")

    with st.sidebar.expander("1. Split documents into chunks", expanded=True):
        st.caption("Smaller chunks are more precise. Larger chunks keep more surrounding context.")
        chunking = st.selectbox(
            "How to split text",
            ["Recursive Character", "Fixed Character", "Sentence"],
            help="Recursive tries natural breakpoints (paragraphs, then sentences). Sentence keeps whole sentences together.",
        )
        chunk_size = st.slider("Target chunk size (characters)", 200, 2400, 800, 100)
        max_overlap = max(0, min(600, chunk_size - 1))
        overlap = st.slider(
            "Overlap between chunks",
            0,
            max_overlap,
            min(150, max_overlap),
            25,
            disabled=chunking == "Sentence",
            help="Repeats text at chunk edges so an idea is not cut in half.",
        )
        if chunking == "Sentence":
            st.caption("Sentence splitting ignores overlap.")

    with st.sidebar.expander("2. Embeddings and index", expanded=False):
        st.caption("Embeddings turn text into vectors so similar passages can be found later.")
        embedding_provider = st.selectbox(
            "Embedding model source",
            ["Sentence Transformers", "OpenAI"],
            key="embedding_provider",
            help="Sentence Transformers runs locally and needs no key.",
        )
        if embedding_provider == "Sentence Transformers":
            embedding_model = st.text_input("Embedding model", "sentence-transformers/all-MiniLM-L6-v2")
            embedding_api_key = None
        else:
            embedding_model = st.text_input("Embedding model", "text-embedding-3-small")
            embedding_api_key = st.text_input(
                "Embedding API key",
                type="password",
                help="Optional if OPENAI_API_KEY is already set.",
            ) or None
        vector_store = st.selectbox(
            "Where to store vectors",
            ["FAISS", "NumPy"],
            help="FAISS is faster for larger collections. NumPy is simpler and fully local.",
        )

    with st.sidebar.expander("3. Retrieval", expanded=True):
        st.caption("These settings apply when you ask a question. They do not rebuild the index.")
        top_k = st.slider("How many chunks to retrieve", 1, 20, 5, help="Higher values give the model more context, but can add noise.")
        threshold = st.slider(
            "Minimum similarity",
            -1.0,
            1.0,
            0.0,
            0.05,
            help="Chunks below this score are dropped. Leave at 0.00 unless results look too weak or too noisy.",
        )
        reranker = st.selectbox(
            "Rerank retrieved chunks",
            ["None", "Cross Encoder"],
            help="A cross-encoder re-scores the shortlist more carefully. Slower, often more precise.",
        )
        reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        if reranker == "Cross Encoder":
            reranker_model = st.text_input("Reranker model", reranker_model)

    generator_provider, generator_model, generator_api_key, base_url, temperature, max_tokens = _render_generator_settings(run_mode)

    st.sidebar.divider()
    st.sidebar.caption(
        "Changing chunking, embeddings, or the vector store rebuilds the index. "
        "Changing the generator, top-k, or threshold does not."
    )

    return {
        "run_mode": run_mode,
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


def _render_generator_settings(run_mode: str):
    with st.sidebar.expander("4. Generator", expanded=True):
        if run_mode == "offline":
            generator_provider = "Extractive (offline)"
            generator_model = GENERATOR_DEFAULTS[generator_provider]
            st.markdown("**Extractive (offline)**")
            st.caption("No model call. The app quotes and stitches the best retrieved passages.")
            generator_api_key = None
            base_url = None
            temperature = 0.2
            max_tokens = 700
            return generator_provider, generator_model, generator_api_key, base_url, temperature, max_tokens

        if run_mode == "local":
            generator_provider = st.selectbox(
                "Local server",
                ["Ollama", "Local LLM (OpenAI-compatible)"],
                key="generator_provider_local",
            )
            default_model = GENERATOR_DEFAULTS[generator_provider]
            generator_model = st.text_input("Model name", default_model, key="generator_model")
            base_url = st.text_input(
                "Server URL",
                value=st.session_state.get("base_url", "http://localhost:11434/v1"),
                key="base_url",
                help="Ollama's OpenAI-compatible endpoint is usually http://localhost:11434/v1",
            ) or None
            with st.expander("Advanced server options"):
                generator_api_key = st.text_input(
                    "API key (usually blank)",
                    type="password",
                    help="Leave empty unless your local server requires authentication.",
                ) or None
        else:
            generator_provider = st.selectbox(
                "Provider",
                CLOUD_GENERATORS,
                key="generator_provider",
            )
            generator_model = st.text_input(
                "Model name",
                GENERATOR_DEFAULTS[generator_provider],
                key="generator_model_cloud",
            )
            generator_api_key = None
            base_url = None
            if generator_provider != "Extractive (offline)":
                generator_api_key = st.text_input(
                    "API key",
                    type="password",
                    help="Optional if the matching environment variable is already set.",
                ) or None
            if generator_provider in {"Ollama", "Local LLM (OpenAI-compatible)", "OpenAI / OpenAI-compatible"}:
                default_url = "http://localhost:11434/v1" if generator_provider in {"Ollama", "Local LLM (OpenAI-compatible)"} else ""
                base_url = st.text_input("Base URL (optional)", value=default_url) or None

        extractive = generator_provider == "Extractive (offline)"
        temperature = st.slider("Creativity (temperature)", 0.0, 1.5, 0.2, 0.1, disabled=extractive)
        max_tokens = st.slider("Maximum answer length (tokens)", 100, 3000, 700, 100, disabled=extractive)
        return generator_provider, generator_model, generator_api_key, base_url, temperature, max_tokens
