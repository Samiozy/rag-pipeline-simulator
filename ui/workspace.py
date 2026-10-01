from pathlib import Path
import hashlib
import tempfile
import streamlit as st

from core.models import Document
from core.preprocessing import clean_documents
from core.generation import create_generator
from core.services.lab import LabComponents, build_components, build_strategy, populate_index
from core.loaders import get_loader
from rag_strategies.base import RAGStrategy
from ui.components import SAMPLE_PATH

INDEX_KEYS = [
    "chunking",
    "chunk_size",
    "overlap",
    "embedding_provider",
    "embedding_model",
    "embedding_api_key",
    "vector_store",
    "reranker_name",
    "reranker_model",
]

SESSION_RESET_KEYS = [
    "documents",
    "chunks",
    "components",
    "strategy",
    "index_fingerprint",
    "doc_source",
    "just_indexed",
    "conversation",
    "upload_fingerprint",
    "pending_question",
    "last_output",
    "compare_result",
]


def index_fingerprint(documents: list[Document], config: dict) -> str:
    digest = hashlib.sha256()
    for document in documents:
        digest.update(document.source.encode())
        digest.update(document.text.encode())
    for key in INDEX_KEYS:
        value = config.get(key)
        digest.update(f"{key}={value is not None}".encode() if str(key).endswith("api_key") else str(value).encode())
    return digest.hexdigest()


def upload_fingerprint(uploaded_files) -> str:
    digest = hashlib.sha256()
    for uploaded in uploaded_files:
        digest.update(uploaded.name.encode())
        digest.update(uploaded.getvalue())
    return digest.hexdigest()


def load_uploaded_files(uploaded_files) -> list[Document]:
    documents = []
    for uploaded in uploaded_files:
        suffix = Path(uploaded.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            temp_path = Path(tmp.name)
        try:
            loader = get_loader(temp_path)
            loaded = loader.load(temp_path)
            for document in loaded:
                document.source = uploaded.name
            documents.extend(loaded)
        finally:
            temp_path.unlink(missing_ok=True)
    cleaned = clean_documents(documents)
    if not cleaned:
        raise ValueError("No text could be extracted from those files.")
    return cleaned


def load_sample_documents() -> list[Document]:
    loader = get_loader(SAMPLE_PATH)
    documents = loader.load(SAMPLE_PATH)
    for document in documents:
        document.source = SAMPLE_PATH.name
    return clean_documents(documents)


def begin_new_document_session() -> None:
    st.session_state["upload_epoch"] = st.session_state.get("upload_epoch", 0) + 1
    for key in SESSION_RESET_KEYS:
        st.session_state.pop(key, None)


def clear_conversation() -> None:
    st.session_state["conversation"] = []
    st.session_state["last_output"] = None
    st.session_state.pop("pending_question", None)


def ensure_index(documents: list[Document], config: dict) -> tuple[LabComponents, list, RAGStrategy]:
    fingerprint = index_fingerprint(documents, config)
    if st.session_state.get("index_fingerprint") != fingerprint:
        with st.spinner("Building the shared search index…"):
            components = build_components(
                chunking=config["chunking"],
                chunk_size=config["chunk_size"],
                overlap=config["overlap"],
                embedding_provider=config["embedding_provider"],
                embedding_model=config["embedding_model"],
                embedding_api_key=config["embedding_api_key"],
                vector_store=config["vector_store"],
                generator_provider=config["generator_provider"],
                generator_model=config["generator_model"],
                generator_api_key=config["generator_api_key"],
                base_url=config["base_url"],
                reranker_name=config["reranker_name"],
                reranker_model=config["reranker_model"],
            )
            chunks = populate_index(components, documents)
            st.session_state.update({
                "index_fingerprint": fingerprint,
                "components": components,
                "chunks": chunks,
                "conversation": [],
                "last_output": None,
                "compare_result": None,
                "just_indexed": True,
            })
            st.rerun()
    components = st.session_state["components"]
    chunks = st.session_state["chunks"]
    components.generator = create_generator(
        config["generator_provider"],
        config["generator_model"],
        config["generator_api_key"],
        config["base_url"],
    )
    if not hasattr(components, "graph_store"):
        st.session_state.pop("index_fingerprint", None)
        st.rerun()
    strategy_key = config["page"] if config["page"] in {
        "naive", "rerank", "hybrid", "graph", "multimodal", "agentic_router", "multi_agent"
    } else "naive"
    strategy = build_strategy(
        strategy_key,
        components,
        retrieval=config["retrieval"],
        rerank=config["rerank"],
        hybrid=config["hybrid"],
        graph=config.get("graph"),
        multimodal=config.get("multimodal"),
        router=config.get("router"),
        multi_agent=config.get("multi_agent"),
    )
    strategy.chunks = chunks
    inner = getattr(strategy, "document_strategy", None)
    if inner is not None:
        inner.chunks = chunks
    return components, chunks, strategy
