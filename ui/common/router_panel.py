import streamlit as st

from rag_strategies.agentic.router.router import RouterDecision


def render_router_decision(decision: RouterDecision, extras: dict) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Selected route", decision.route)
    c2.metric("Confidence", f"{decision.confidence:.2f}")
    c3.metric("Tool", decision.tool or "—")
    st.write(decision.reason)
    st.caption("Available routes: " + ", ".join(extras.get("available_routes") or ["rag", "memory", "tool", "direct"]))
    st.caption("This is a structured decision summary. Hidden model chain-of-thought is not stored.")
