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
from rag.pipeline import RAGPipeline
from rag.prompts import DEFAULT_SYSTEM_PROMPT
from rag.evaluation import retrieval_summary
from ui.sidebar import render_sidebar
from ui.document_view import render_documents
from ui.chunk_view import render_chunks
from ui.retrieval_view import render_retrieval

load_dotenv()
st.set_page_config(page_title="RAG Pipeline Simulator", page_icon="🧪", layout="wide")


@st.cache_resource(show_spinner=False)
def cached_embedder(provider: str, model_name: str, api_key: Optional[str]):
    return create_embedder(provider, model_name, api_key)


def file_fingerprint(files, config: dict) -> str:
    digest = hashlib.sha256()
    for f in files:
        digest.update(f.name.encode())
        digest.update(f.getvalue())
    for key in ["chunking", "chunk_size", "overlap", "embedding_provider", "embedding_model", "vector_store", "reranker", "reranker_model", "generator_provider", "generator_model", "base_url"]:
        digest.update(str(config[key]).encode())
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
    if config["reranker"] == "Cross Encoder":
        reranker = CrossEncoderReranker(config["reranker_model"])
    else:
        reranker = NoOpReranker()
    return RAGPipeline(chunker, embedder, vector_store, generator, reranker)


st.title("🧪 RAG Pipeline Simulator")
st.caption("Inspect, configure, and compare the major stages of retrieval-augmented generation without coupling the application to one LLM provider.")

config = render_sidebar()

with st.expander("Pipeline architecture", expanded=False):
    st.code("Document → Clean → Chunk → Embed → Vector Store → Retrieve → Rerank → Prompt → Generate → Inspect", language=None)

uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf", "txt", "md", "docx"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload one or more PDF, TXT, Markdown, or DOCX files to begin.")
    st.stop()

fingerprint = file_fingerprint(uploaded_files, config)
if st.session_state.get("fingerprint") != fingerprint:
    with st.spinner("Building retrieval index..."):
        try:
            documents = load_uploaded_files(uploaded_files)
            pipeline = build_pipeline(config)
            chunks = pipeline.ingest(documents)
            st.session_state.update({
                "fingerprint": fingerprint,
                "documents": documents,
                "pipeline": pipeline,
                "chunks": chunks,
                "last_output": None,
            })
        except Exception as exc:
            st.error(f"Indexing failed: {exc}")
            st.stop()

pipeline = st.session_state["pipeline"]
documents = st.session_state["documents"]
chunks = st.session_state["chunks"]

doc_tab, chunk_tab, query_tab, prompt_tab, metrics_tab = st.tabs([
    "1 · Document", "2 · Chunks", "3 · Retrieval & Answer", "4 · Prompt Inspector", "5 · Metrics"
])

with doc_tab:
    render_documents(documents)

with chunk_tab:
    render_chunks(chunks)

with query_tab:
    question = st.text_area("Question", placeholder="Ask a question grounded in the uploaded documents...")
    system_prompt = st.text_area("System / RAG instruction", value=DEFAULT_SYSTEM_PROMPT, height=120)
    if st.button("Run RAG", type="primary", disabled=not question.strip()):
        try:
            with st.spinner("Retrieving context and generating answer..."):
                output = pipeline.query(
                    question.strip(),
                    top_k=config["top_k"],
                    threshold=config["threshold"],
                    temperature=config["temperature"],
                    max_tokens=config["max_tokens"],
                    system_prompt=system_prompt,
                )
                st.session_state["last_output"] = output
        except Exception as exc:
            st.error(f"Query failed: {exc}")

    output = st.session_state.get("last_output")
    if output:
        st.subheader("Generated answer")
        st.write(output.answer)
        st.subheader("Retrieved chunks")
        render_retrieval(output.retrieved)

with prompt_tab:
    output = st.session_state.get("last_output")
    if output:
        st.code(output.prompt, language=None)
    else:
        st.info("Run a query to inspect the exact prompt sent to the generator.")

with metrics_tab:
    output = st.session_state.get("last_output")
    if output:
        summary = retrieval_summary(output.retrieved)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Retrieved", summary["retrieved"])
        c2.metric("Average score", f"{summary['avg_score']:.4f}")
        c3.metric("Retrieval latency", f"{output.retrieval_ms:.1f} ms")
        c4.metric("Generation latency", f"{output.generation_ms:.1f} ms")
        st.caption("Similarity scores are directly comparable within a run; cross-encoder reranker scores may use a different numerical scale.")
    else:
        st.info("Run a query to populate metrics.")
