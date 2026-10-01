import streamlit as st
from dotenv import load_dotenv

from ui.components import render_empty_state, render_header, render_status_banner, render_stepper, current_stage
from ui.pages.compare import render_compare
from ui.pages.laboratory import render_laboratory
from ui.pages.overview import render_overview
from ui.sidebar import render_navigation, render_sidebar
from ui.styles import inject_styles
from ui.workspace import (
    begin_new_document_session,
    clear_conversation,
    ensure_index,
    load_sample_documents,
    load_uploaded_files,
    upload_fingerprint,
)

load_dotenv()
st.set_page_config(page_title="RAG Systems Laboratory", page_icon="🧪", layout="wide")
inject_styles()

if "upload_epoch" not in st.session_state:
    st.session_state["upload_epoch"] = 0
if "conversation" not in st.session_state:
    st.session_state["conversation"] = []

render_header()
if "pending_lab_page" in st.session_state:
    st.session_state["lab_page"] = st.session_state.pop("pending_lab_page")
page = render_navigation()
config = render_sidebar(page)
has_documents = bool(st.session_state.get("documents"))
render_stepper(current_stage(has_documents, bool(st.session_state.get("conversation"))))

uploader_kwargs = dict(
    label="Upload PDF, TXT, Markdown, DOCX, or images",
    type=["pdf", "txt", "md", "docx", "png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
    key=f"doc_uploader_{st.session_state['upload_epoch']}",
)

if not has_documents:
    if page == "overview":
        render_overview()
    if render_empty_state():
        st.session_state["documents"] = load_sample_documents()
        st.session_state["doc_source"] = "sample"
        if st.session_state.get("lab_page") == "overview":
            st.session_state["pending_lab_page"] = "naive"
        clear_conversation()
        st.rerun()
    uploaded_files = st.file_uploader(**uploader_kwargs)
else:
    with st.expander("Start a new document session"):
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
    if page != "overview":
        st.info("Add a document or try the sample to open this architecture.")
    st.stop()

if page == "overview":
    render_overview()
    st.stop()

try:
    components, chunks, strategy = ensure_index(st.session_state["documents"], config)
except Exception as exc:
    st.error(f"Indexing failed: {exc}")
    st.stop()

if st.session_state.get("just_indexed"):
    st.toast("Index ready. Ask a question or open Compare.")
    st.session_state["just_indexed"] = False

documents = st.session_state["documents"]
config["conversation_turns"] = len(st.session_state.get("conversation") or [])

tools = st.columns([3.2, 1, 1])
with tools[0]:
    render_status_banner(documents, chunks, {**config, "generator_provider": config["generator_provider"], "generator_model": config["generator_model"], "chunking": config["chunking"], "chunk_size": config["chunk_size"], "embedding_provider": config["embedding_provider"], "vector_store": config["vector_store"], "top_k": config["retrieval"].top_k, "threshold": config["retrieval"].similarity_threshold or 0.0})
with tools[1]:
    if st.button("New conversation", use_container_width=True, disabled=not st.session_state.get("conversation")):
        clear_conversation()
        st.rerun()
with tools[2]:
    if st.button("New document session", use_container_width=True):
        begin_new_document_session()
        st.rerun()

if page == "compare":
    render_compare(documents, chunks, components, config)
else:
    render_laboratory(page, documents, chunks, strategy, config)
