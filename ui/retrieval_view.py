import pandas as pd
import streamlit as st
from models import RetrievalResult


def render_retrieval(results: list[RetrievalResult]) -> None:
    if not results:
        st.warning("No chunks passed the retrieval threshold.")
        return
    rows = []
    for r in results:
        rows.append({
            "rank": r.rank,
            "score": round(r.score, 4),
            "source": r.chunk.source,
            "page": r.chunk.metadata.get("page", ""),
            "preview": r.chunk.text[:260].replace("\n", " "),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    for r in results:
        page = r.chunk.metadata.get("page")
        label = f"#{r.rank} · score {r.score:.4f} · {r.chunk.source}" + (f" · page {page}" if page else "")
        with st.expander(label):
            st.write(r.chunk.text)
