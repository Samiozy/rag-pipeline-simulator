from typing import Optional

import streamlit as st

from rag.pipeline import QueryOutput
from ui.retrieval_view import render_retrieval


def render_transcript(turns: list[QueryOutput]) -> None:
    if not turns:
        st.info("Ask a first question, then follow up in the same conversation. Retrieval still runs on every turn.")
        return

    for index, turn in enumerate(turns, start=1):
        with st.chat_message("user"):
            st.markdown(turn.question)
        with st.chat_message("assistant"):
            st.markdown(turn.answer)
            caption = f"Turn {index} · {turn.retrieval_ms:.0f} ms retrieve · {turn.generation_ms:.0f} ms generate"
            if turn.retrieval_query and turn.retrieval_query != turn.question:
                caption += " · follow-up retrieval used prior questions"
            st.caption(caption)
            with st.expander("Evidence for this turn"):
                if turn.retrieval_query != turn.question:
                    st.caption("Search query sent to the index")
                    st.code(turn.retrieval_query, language=None)
                render_retrieval(turn.retrieved)


def selected_turn(turns: list[QueryOutput], widget_key: str) -> Optional[QueryOutput]:
    if not turns:
        return None
    if len(turns) == 1:
        return turns[-1]
    labels = [f"Turn {i}: {turn.question[:72]}" for i, turn in enumerate(turns, start=1)]
    chosen = st.selectbox("Inspect a turn", labels, index=len(labels) - 1, key=widget_key)
    return turns[labels.index(chosen)]
