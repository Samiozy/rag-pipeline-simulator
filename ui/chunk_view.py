import pandas as pd
import streamlit as st
from models import Chunk


def render_chunks(chunks: list[Chunk]) -> None:
    if not chunks:
        st.info("No chunks yet. Upload a document or load the sample to build the index.")
        return

    lengths = [len(chunk.text) for chunk in chunks]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Chunks", len(chunks))
    c2.metric("Average length", f"{int(sum(lengths) / len(lengths)):,} chars")
    c3.metric("Shortest", f"{min(lengths):,} chars")
    c4.metric("Longest", f"{max(lengths):,} chars")
    st.caption("If chunks look too fragmented or too large, change the split settings in the sidebar. That rebuilds the index.")

    rows = []
    options = []
    for i, chunk in enumerate(chunks, 1):
        page = chunk.metadata.get("page", "")
        rows.append({
            "#": i,
            "source": chunk.source,
            "page": page,
            "characters": len(chunk.text),
            "preview": chunk.text[:180].replace("\n", " "),
        })
        page_bit = f" · page {page}" if page else ""
        options.append(f"{i} — {chunk.source}{page_bit} ({len(chunk.text)} chars)")

    table, inspector = st.columns([1.15, 0.85])
    with table:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=360)
    with inspector:
        selected_label = st.selectbox("Read a chunk", options)
        selected_index = options.index(selected_label)
        chunk = chunks[selected_index]
        st.caption(f"{len(chunk.text):,} characters")
        st.code(chunk.text, language=None)
