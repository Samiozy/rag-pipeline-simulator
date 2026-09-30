from pathlib import Path
import hashlib
import tempfile
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

from rag.ingestion import get_loader
from rag.preprocessing import clean_documents
from rag.chunking import make_chunker
from rag.embeddings import create_embedder
from rag.vectorstores import create_vector_store
from rag.generation import create_generator
from rag.retrieval import NoOpReranker, CrossEncoderReranker
from rag.pipeline import RAGPipeline, QueryOutput
from rag.prompts import DEFAULT_SYSTEM_PROMPT
from ui.sidebar import render_sidebar
from ui.document_view import render_documents
from ui.chunk_view import render_chunks
from ui.prompt_view import render_prompt
from ui.metrics_view import render_metrics
from ui.chat_view import render_transcript, selected_turn
from ui.styles import inject_styles
from ui.components import (
    SAMPLE_PATH,
    current_stage,
    render_example_questions,
    render_header,
    render_status_banner,
    render_stepper,
    render_empty_state,
)

load_dotenv()
st.set_page_config(page_title="RAG Pipeline Simulator", page_icon="🧪", layout="wide")
inject_styles()

INDEX_KEYS = [
    "chunking",
    "chunk_size",
    "overlap",
    "embedding_provider",
    "embedding_model",
    "embedding_api_key",
    "vector_store",
]
SESSION_RESET_KEYS = [
    "documents",
    "pipeline",
    "chunks",
    "last_output",
    "index_fingerprint",
    "doc_source",
    "just_indexed",
    "conversation",
    "upload_fingerprint",
    "pending_question",
]


@st.cache_resource(show_spinner=False)
def cached_embedder(provider: str, model_name: str, api_key: Optional[str]):
    return create_embedder(provider, model_name, api_key)


@st.cache_resource(show_spinner=False)
def cached_reranker(model_name: str):
    return CrossEncoderReranker(model_name)


def index_fingerprint(documents, config: dict) -> str:
    digest = hashlib.sha256()
    for document in documents:
        digest.update(document.source.encode())
        digest.update(document.text.encode())
    for key in INDEX_KEYS:
        value = config[key]
        digest.update(f"{key}={value is not None}".encode() if key.endswith("api_key") else str(value).encode())
    return digest.hexdigest()


def upload_fingerprint(uploaded_files) -> str:
    digest = hashlib.sha256()
    for uploaded in uploaded_files:
        digest.update(uploaded.name.encode())
        digest.update(uploaded.getvalue())
    return digest.hexdigest()


def load_uploaded_files(uploaded_files):
    documents = []
    for uploaded in uploaded_files:
        suffix = Path(uploaded.name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            temp_path = Path(tmp.name)
        try:
            loader = get_loader(temp_path)
            loaded = loader.load(temp_path)
            for doc in loaded:
                doc.source = uploaded.name
            documents.extend(loaded)
        finally:
            temp_path.unlink(missing_ok=True)
    return clean_documents(documents)


def load_sample_documents():
    loader = get_loader(SAMPLE_PATH)
    documents = loader.load(SAMPLE_PATH)
    for document in documents:
        document.source = SAMPLE_PATH.name
    return clean_documents(documents)


def build_pipeline(config: dict) -> RAGPipeline:
    chunker = make_chunker(config["chunking"], config["chunk_size"], config["overlap"])
    embedder = cached_embedder(config["embedding_provider"], config["embedding_model"], config["embedding_api_key"])
    vector_store = create_vector_store(config["vector_store"])
    generator = create_generator(
        config["generator_provider"],
        config["generator_model"],
        config["generator_api_key"],
        config["base_url"],
    )
    reranker = (
        cached_reranker(config["reranker_model"])
        if config["reranker"] == "Cross Encoder"
        else NoOpReranker()
    )
    return RAGPipeline(chunker, embedder, vector_store, generator, reranker)


def apply_query_components(pipeline: RAGPipeline, config: dict) -> None:
    pipeline.generator = create_generator(
        config["generator_provider"],
        config["generator_model"],
        config["generator_api_key"],
        config["base_url"],
    )
    pipeline.reranker = (
        cached_reranker(config["reranker_model"])
        if config["reranker"] == "Cross Encoder"
        else NoOpReranker()
    )


def begin_new_document_session() -> None:
    st.session_state["upload_epoch"] = st.session_state.get("upload_epoch", 0) + 1
    for key in SESSION_RESET_KEYS:
        st.session_state.pop(key, None)


def clear_conversation() -> None:
    st.session_state["conversation"] = []
    st.session_state["last_output"] = None
    st.session_state.pop("pending_question", None)


def conversation_history(turns: list[QueryOutput]) -> list[tuple[str, str]]:
    return [(turn.question, turn.answer) for turn in turns]


def run_query(pipeline: RAGPipeline, question: str, config: dict, system_prompt: str) -> None:
    turns = list(st.session_state.get("conversation") or [])
    output = pipeline.query(
        question.strip(),
        top_k=config["top_k"],
        threshold=config["threshold"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
        system_prompt=system_prompt,
        history=conversation_history(turns),
    )
    turns.append(output)
    st.session_state["conversation"] = turns
    st.session_state["last_output"] = output


if "upload_epoch" not in st.session_state:
    st.session_state["upload_epoch"] = 0
if "conversation" not in st.session_state:
    st.session_state["conversation"] = []

render_header()
config = render_sidebar()
has_documents = bool(st.session_state.get("documents"))
render_stepper(current_stage(has_documents, bool(st.session_state.get("conversation"))))

uploader_kwargs = dict(
    label="Upload PDF, TXT, Markdown, or DOCX files",
    type=["pdf", "txt", "md", "docx"],
    accept_multiple_files=True,
    key=f"doc_uploader_{st.session_state['upload_epoch']}",
    help="You can add several files. They are combined into one searchable collection.",
)

if not has_documents:
    if render_empty_state():
        st.session_state["documents"] = load_sample_documents()
        st.session_state["doc_source"] = "sample"
        clear_conversation()
        st.rerun()
    uploaded_files = st.file_uploader(**uploader_kwargs)
else:
    with st.expander("Start a new document session"):
        st.caption("Upload replacement files to rebuild the index and clear the current chat. Or use New document session to return to an empty uploader.")
        uploaded_files = st.file_uploader(**uploader_kwargs)

if uploaded_files:
    incoming = upload_fingerprint(uploaded_files)
    if incoming != st.session_state.get("upload_fingerprint"):
        try:
            st.session_state["documents"] = load_uploaded_files(uploaded_files)
            st.session_state["doc_source"] = "upload"
            st.session_state["upload_fingerprint"] = incoming
            clear_conversation()
        except Exception as exc:
            st.error(f"Could not read those files: {exc}")
            st.stop()

if not st.session_state.get("documents"):
    st.stop()

if st.session_state.get("just_indexed"):
    st.toast("Index ready. Ask a question, then follow up in the same chat.")
    st.session_state["just_indexed"] = False

documents = st.session_state["documents"]
fingerprint = index_fingerprint(documents, config)
if st.session_state.get("index_fingerprint") != fingerprint:
    with st.spinner("Building the search index from your documents…"):
        try:
            pipeline = build_pipeline(config)
            chunks = pipeline.ingest(documents)
            st.session_state.update({
                "index_fingerprint": fingerprint,
                "pipeline": pipeline,
                "chunks": chunks,
                "last_output": None,
                "conversation": [],
                "just_indexed": True,
            })
            st.rerun()
        except Exception as exc:
            st.error(f"Indexing failed: {exc}")
            st.stop()
else:
    try:
        apply_query_components(st.session_state["pipeline"], config)
    except Exception as exc:
        st.error(f"Could not update the generator: {exc}")
        st.stop()

pipeline = st.session_state["pipeline"]
documents = st.session_state["documents"]
chunks = st.session_state["chunks"]
config["conversation_turns"] = len(st.session_state.get("conversation") or [])

tools = st.columns([3.2, 1, 1])
with tools[0]:
    render_status_banner(documents, chunks, config)
with tools[1]:
    if st.button("New conversation", use_container_width=True, disabled=not st.session_state.get("conversation")):
        clear_conversation()
        st.rerun()
with tools[2]:
    if st.button("New document session", use_container_width=True):
        begin_new_document_session()
        st.rerun()

if st.session_state.get("doc_source") == "sample":
    st.caption("You are exploring with the built-in sample. Start a new document session to upload your own files.")

doc_tab, chunk_tab, query_tab, prompt_tab, metrics_tab = st.tabs([
    "Documents",
    "Chunks",
    "Ask",
    "Prompt",
    "Metrics",
])

with doc_tab:
    st.caption("This is the cleaned text the pipeline will search — not the generated answer.")
    render_documents(documents)

with chunk_tab:
    render_chunks(chunks)

with query_tab:
    conversation = st.session_state.get("conversation") or []
    using_sample = any(document.source == SAMPLE_PATH.name for document in documents)
    st.caption("Each follow-up is retrieved against the documents, with prior questions added so references like “that” still search well.")
    render_transcript(conversation)
    if using_sample and not conversation:
        render_example_questions()

    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
    with st.expander("Instructions for the generator", expanded=False):
        system_prompt = st.text_area(
            "System / RAG instruction",
            height=120,
            key="system_prompt",
            label_visibility="collapsed",
        )
        st.caption("These instructions are prepended to the retrieved context. They are not used to search the documents.")

    incoming = st.session_state.get("pending_question")
    chat_prompt = st.chat_input("Ask a question, or follow up on the last answer…")
    if chat_prompt:
        incoming = chat_prompt.strip()

    if incoming:
        try:
            with st.spinner("Finding relevant passages and composing an answer…"):
                run_query(pipeline, incoming, config, st.session_state.system_prompt)
            st.session_state.pop("pending_question", None)
            st.rerun()
        except Exception as exc:
            st.error(f"The query failed: {exc}")

with prompt_tab:
    turns = st.session_state.get("conversation") or []
    render_prompt(selected_turn(turns, "prompt_turn"))

with metrics_tab:
    turns = st.session_state.get("conversation") or []
    render_metrics(selected_turn(turns, "metrics_turn"), turns)
