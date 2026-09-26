import pandas as pd
import streamlit as st
from models import Chunk


def render_chunks(chunks: list[Chunk]) -> None:
    if not chunks:
        st.info("No chunks generated yet.")
        return
    lengths = [len(c.text) for c in chunks]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Chunks", len(chunks))
    c2.metric("Avg chars", int(sum(lengths) / len(lengths)))
    c3.metric("Min chars", min(lengths))
    c4.metric("Max chars", max(lengths))

    rows = []
    for i, chunk in enumerate(chunks, 1):
        rows.append({
            "#": i,
            "source": chunk.source,
            "page": chunk.metadata.get("page", ""),
            "characters": len(chunk.text),
            "preview": chunk.text[:220].replace("\n", " "),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    selected = st.number_input("Inspect chunk", min_value=1, max_value=len(chunks), value=1)
    chunk = chunks[int(selected) - 1]
    st.code(chunk.text, language=None)
