from html import escape
from pathlib import Path
from typing import Optional

import streamlit as st

from models import Document, Chunk

SAMPLE_PATH = Path(__file__).resolve().parents[1] / "data" / "sample.txt"

SAMPLE_QUESTIONS = [
    "Why should retrieval be evaluated independently from generation?",
    "How does chunk size affect retrieval quality?",
    "What is chunk overlap used for?",
]

STEPS = [
    ("Add documents", "upload"),
    ("Inspect chunks", "chunks"),
    ("Ask a question", "ask"),
    ("Review evidence", "inspect"),
]


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-kicker">Local RAG laboratory</div>
        <h1 class="hero-title">See how a question becomes an answer</h1>
        <p class="hero-copy">
            Upload documents, watch them get chunked and retrieved, then inspect the exact
            context sent to the generator. Start offline — no API key required.
        </p>
        """,
        unsafe_allow_html=True,
    )


def render_stepper(current: str) -> None:
    order = [key for _, key in STEPS]
    current_idx = order.index(current) if current in order else 0
    chips = []
    for index, (label, key) in enumerate(STEPS, start=1):
        css = "is-current" if key == current else "is-done" if index - 1 < current_idx else ""
        chips.append(
            f'<div class="step {css}"><b>{index}</b>{escape(label)}</div>'
        )
    st.markdown(f'<div class="stepper">{"".join(chips)}</div>', unsafe_allow_html=True)


def _unique_sources(documents: list[Document]) -> list[str]:
    seen = []
    for doc in documents:
        if doc.source not in seen:
            seen.append(doc.source)
    return seen


def render_status_banner(documents: list[Document], chunks: list[Chunk], config: dict) -> None:
    sources = _unique_sources(documents)
    generator = config["generator_provider"]
    if generator == "Extractive (offline)":
        generator_label = "Offline extractive answers"
    else:
        generator_label = f"{generator} · {config['generator_model']}"

    chips = [
        f'<span class="chip"><strong>{len(sources)}</strong> file{"s" if len(sources) != 1 else ""}</span>',
        f'<span class="chip"><strong>{len(chunks)}</strong> chunks</span>',
        f'<span class="chip">{escape(config["chunking"])} · {config["chunk_size"]} chars</span>',
        f'<span class="chip">{escape(config["embedding_provider"])} → {escape(config["vector_store"])}</span>',
        f'<span class="chip">{escape(generator_label)}</span>',
        f'<span class="chip">Top {config["top_k"]} · threshold {config["threshold"]:.2f}</span>',
    ]
    turns = config.get("conversation_turns", 0)
    if turns:
        chips.append(f'<span class="chip"><strong>{turns}</strong> chat turn{"s" if turns != 1 else ""}</span>')
    st.markdown(f'<div class="status-bar">{"".join(chips)}</div>', unsafe_allow_html=True)
    if sources:
        st.caption("Indexed files: " + ", ".join(sources))


def render_empty_state() -> bool:
    st.markdown(
        """
        <div class="empty-card">
            <h3>Get a first run in under a minute</h3>
            <p>You can explore the whole pipeline with the built-in sample, then start a new document session whenever you want.</p>
            <div class="empty-steps">
                <div class="empty-step"><span>Step 1</span>Add a document</div>
                <div class="empty-step"><span>Step 2</span>Ask, then follow up</div>
                <div class="empty-step"><span>Step 3</span>Inspect retrieved chunks, the prompt, and latency</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    left, right = st.columns([1, 1])
    with left:
        use_sample = st.button("Try the sample document", type="primary", use_container_width=True)
    with right:
        st.caption("Or upload a PDF, TXT, Markdown, or DOCX file below. Offline mode works without an API key.")
    return use_sample


def current_stage(has_documents: bool, has_output: bool) -> str:
    if not has_documents:
        return "upload"
    if not has_output:
        return "ask"
    return "inspect"


def score_bar_html(score: float, low: float, high: float) -> str:
    if high <= low:
        width = 100 if score >= low else 0
    else:
        width = max(4, min(100, int(((score - low) / (high - low)) * 100)))
    return (
        f'<div class="score-track"><div class="score-fill" style="width:{width}%"></div></div>'
        f'<small>Similarity {score:.3f}</small>'
    )


def set_question(question: str) -> None:
    st.session_state["pending_question"] = question


def render_example_questions(questions: Optional[list[str]] = None) -> None:
    options = questions or SAMPLE_QUESTIONS
    st.caption("Try a starter question")
    columns = st.columns(len(options))
    for i, question in enumerate(options):
        columns[i].button(
            question,
            key=f"example_q_{i}",
            on_click=set_question,
            args=(question,),
            use_container_width=True,
        )
