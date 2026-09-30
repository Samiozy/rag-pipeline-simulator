from typing import Optional

import streamlit as st

from rag.pipeline import QueryOutput


def _split_prompt(prompt: str) -> dict[str, str]:
    parts = {"instructions": "", "conversation": "", "context": "", "question": ""}
    rest = prompt
    if "\n\nCONVERSATION\n" in rest:
        parts["instructions"], rest = rest.split("\n\nCONVERSATION\n", 1)
        if "\n\nCONTEXT\n" in rest:
            parts["conversation"], rest = rest.split("\n\nCONTEXT\n", 1)
        else:
            parts["conversation"] = rest
            rest = ""
    elif "\n\nCONTEXT\n" in rest:
        parts["instructions"], rest = rest.split("\n\nCONTEXT\n", 1)
    else:
        parts["instructions"] = rest
        rest = ""

    if rest:
        if "\n\nQUESTION\n" in rest:
            parts["context"], rest = rest.split("\n\nQUESTION\n", 1)
        else:
            parts["context"] = rest
            rest = ""
    if rest:
        if "\n\nANSWER" in rest:
            parts["question"], _ = rest.split("\n\nANSWER", 1)
        else:
            parts["question"] = rest

    for key, value in parts.items():
        parts[key] = value.strip()
    return parts


def render_prompt(output: Optional[QueryOutput]) -> None:
    if not output:
        st.info("Ask a question in the Ask tab to see the exact prompt the generator received.")
        return

    st.caption("This is the full message constructed from your instructions, conversation so far, retrieved chunks, and the latest question.")
    structured, raw = st.tabs(["Readable view", "Raw prompt"])

    parts = _split_prompt(output.prompt)
    with structured:
        st.markdown("**Instructions**")
        st.write(parts["instructions"] or "—")
        if parts["conversation"]:
            st.markdown("**Conversation so far**")
            st.text(parts["conversation"])
        st.markdown("**Retrieved context**")
        st.text(parts["context"] or "No context retrieved.")
        st.markdown("**Question**")
        st.write(parts["question"] or output.question)
        if output.retrieval_query and output.retrieval_query != output.question:
            st.markdown("**Query used for retrieval**")
            st.code(output.retrieval_query, language=None)

    with raw:
        st.code(output.prompt, language=None)
        st.caption(f"{len(output.prompt):,} characters sent to the generator.")
