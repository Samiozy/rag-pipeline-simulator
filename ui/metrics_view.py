from typing import Optional

import pandas as pd
import streamlit as st

from core.evaluation import retrieval_summary
from core.models import GenerationResult


def render_metrics(output: Optional[GenerationResult], conversation: Optional[list[GenerationResult]] = None) -> None:
    turns = conversation or []
    if not output and not turns:
        st.info("Ask a question to populate retrieval and latency numbers for that run.")
        return

    if len(turns) > 1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Turns", len(turns))
        c2.metric("Total retrieve", f"{sum(turn.retrieval_ms for turn in turns):.0f} ms")
        c3.metric("Total generate", f"{sum(turn.generation_ms for turn in turns):.0f} ms")

    if not output:
        return

    summary = retrieval_summary(output.retrieved)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Chunks used", summary["retrieved"])
    c2.metric("Average score", f"{summary['avg_score']:.3f}")
    c3.metric("Find time", f"{output.retrieval_ms:.0f} ms")
    c4.metric("Answer time", f"{output.generation_ms:.0f} ms")
    c5.metric("Context size", f"{output.context_chars:,} chars")
    st.caption(f"Estimated context tokens: {output.context_tokens_est:,}. These are measurements, not quality scores.")

    if output.retrieved:
        chart = pd.DataFrame(
            {
                "chunk": [f"#{result.rank}" for result in output.retrieved],
                "score": [result.score for result in output.retrieved],
            }
        )
        st.bar_chart(chart, x="chunk", y="score", height=220)
    else:
        st.warning("No chunks were retrieved, so latency is the only useful signal from this run.")
