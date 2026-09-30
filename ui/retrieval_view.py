import pandas as pd
import streamlit as st
from models import RetrievalResult

from ui.components import score_bar_html


def render_retrieval(results: list[RetrievalResult]) -> None:
    if not results:
        st.warning(
            "Nothing passed the similarity threshold. Lower “Minimum similarity” in the sidebar, "
            "ask a more specific question, or check that the documents actually mention this topic."
        )
        return

    scores = [result.score for result in results]
    low, high = min(scores), max(scores)

    st.caption("These are the passages the generator was allowed to use, ranked by similarity.")
    for result in results:
        page = result.chunk.metadata.get("page")
        location = f"{result.chunk.source}" + (f" · page {page}" if page else "")
        title = f"#{result.rank}  {location}"
        with st.expander(title, expanded=result.rank == 1):
            st.markdown(score_bar_html(result.score, low, high), unsafe_allow_html=True)
            st.write(result.chunk.text)

    with st.expander("View as a table"):
        rows = [
            {
                "rank": result.rank,
                "score": round(result.score, 4),
                "source": result.chunk.source,
                "page": result.chunk.metadata.get("page", ""),
                "preview": result.chunk.text[:220].replace("\n", " "),
            }
            for result in results
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
