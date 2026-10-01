import streamlit as st


def render_overview() -> None:
    st.markdown("### A laboratory for RAG architectures")
    st.write(
        "This repository is one shared toolkit with independent retrieval strategies. "
        "Upload a corpus once, then inspect how each design changes retrieved context, ranking, latency, and answers."
    )
    st.markdown("#### Implemented strategies")
    st.markdown(
        """
- **Naive RAG** — chunk, embed, vector search, prompt, generate
- **Retrieve-and-Rerank** — wider first pass, then a reranker
- **Hybrid RAG** — dense + BM25 + rank fusion
- **Graph RAG** — entities, relationships, traversal
- **Multimodal RAG** — text and image-caption items, modality labels
- **Agentic RAG — Router** — structured route: rag / memory / tool / direct
- **Multi-Agent RAG** — coordinator, retrieval agents, synthesis
- **Compare** — same query, two strategies, overlap and latency
"""
    )
    st.markdown("#### Shared infrastructure")
    st.code("Query → Strategy registry → Naive | Rerank | Hybrid | Graph | Multimodal | Router | Multi-Agent", language=None)
    st.caption("Open a strategy page, add the sample document, ask a question, then use Compare.")
