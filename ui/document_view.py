import streamlit as st
from models import Document


def render_documents(documents: list[Document]) -> None:
    total_chars = sum(len(d.text) for d in documents)
    c1, c2, c3 = st.columns(3)
    c1.metric("Document units", len(documents))
    c2.metric("Characters", f"{total_chars:,}")
    c3.metric("Estimated words", f"{sum(len(d.text.split()) for d in documents):,}")
    for i, doc in enumerate(documents[:20], 1):
        page = doc.metadata.get("page")
        label = f"{doc.source} — page {page}" if page else doc.source
        with st.expander(f"{i}. {label}"):
            st.text(doc.text[:6000])
