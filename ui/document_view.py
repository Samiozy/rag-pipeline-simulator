import streamlit as st
from models import Document


def render_documents(documents: list[Document]) -> None:
    if not documents:
        st.info("No documents loaded yet.")
        return

    sources = {doc.source for doc in documents}
    total_chars = sum(len(doc.text) for doc in documents)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Files", len(sources))
    c2.metric("Pages / sections", len(documents))
    c3.metric("Characters", f"{total_chars:,}")
    c4.metric("Words", f"{sum(len(doc.text.split()) for doc in documents):,}")
    st.caption("PDFs are shown page by page so you can see exactly what was ingested.")

    query = st.text_input("Filter by file name or text", placeholder="Type to filter…")
    filtered = documents
    if query.strip():
        needle = query.strip().lower()
        filtered = [
            doc for doc in documents
            if needle in doc.source.lower() or needle in doc.text.lower()
        ]
        st.caption(f"Showing {len(filtered)} of {len(documents)} sections")

    if not filtered:
        st.warning("No sections match that filter.")
        return

    limit = 25
    for i, doc in enumerate(filtered[:limit], 1):
        page = doc.metadata.get("page")
        label = f"{doc.source} · page {page}" if page else doc.source
        preview = doc.text.strip().splitlines()[0][:90] if doc.text.strip() else "Empty section"
        with st.expander(f"{i}. {label} — {len(doc.text):,} characters"):
            st.caption(preview)
            st.text(doc.text[:8000])
            if len(doc.text) > 8000:
                st.caption("Showing the first 8,000 characters of this section.")

    if len(filtered) > limit:
        st.info(f"Showing the first {limit} sections. Use the filter to narrow the list.")
