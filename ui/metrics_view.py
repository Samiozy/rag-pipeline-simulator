from typing import Optional

import pandas as pd
import streamlit as st

from rag.evaluation import retrieval_summary
from rag.pipeline import QueryOutput


def render_metrics(output: Optional[QueryOutput], conversation: Optional[list[QueryOutput]] = None) -> None:
    turns = conversation or []
    if not output and not turns:
        st.info("Ask a question to populate retrieval and latency numbers for that run.")
        return

    if len(turns) > 1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Turns", len(turns))
        c2.metric("Total retrieve", f"{sum(turn.retrieval_ms for turn in turns):.0f} ms")
        c3.metric("Total generate", f"{sum(turn.generation_ms for turn in turns):.0f} ms")
        st.caption("Figures below are for the selected turn. Follow-ups search with prior questions included.")

    if not output:
        return

    summary = retrieval_summary(output.retrieved)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Chunks used", summary["retrieved"], help="How many passages were sent to the generator.")
    c2.metric("Average score", f"{summary['avg_score']:.3f}", help="Mean similarity of the kept chunks.")
    c3.metric("Find time", f"{output.retrieval_ms:.0f} ms", help="Embedding the question, searching, and optional reranking.")
    c4.metric("Answer time", f"{output.generation_ms:.0f} ms", help="Time spent in the generator after retrieval.")

    if output.retrieved:
        chart = pd.DataFrame(
            {
                "chunk": [f"#{result.rank}" for result in output.retrieved],
                "score": [result.score for result in output.retrieved],
            }
        )
        st.bar_chart(chart, x="chunk", y="score", height=220)
        best = max(output.retrieved, key=lambda result: result.score)
        st.caption(
            f"Strongest match: #{best.rank} from {best.chunk.source} at {best.score:.3f}. "
            "Scores are comparable within this run. A cross-encoder may use a different scale than the vector search."
        )
    else:
        st.warning("No chunks were retrieved, so latency is the only useful signal from this run.")
