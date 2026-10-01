import pandas as pd
import streamlit as st

from core.models import TraceEvent


def render_traces(events: list[TraceEvent]) -> None:
    if not events:
        st.info("Run a query to see stage order and timing. Hidden model chain-of-thought is never stored.")
        return
    rows = [
        {
            "order": index,
            "stage": event.stage,
            "input": event.input_summary,
            "output": event.output_summary,
            "ms": round(event.duration_ms, 1),
        }
        for index, event in enumerate(events, 1)
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
