import streamlit as st

APP_CSS = """
<style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 6.5rem;
        max-width: 1280px;
    }

    [data-testid="stSidebar"] {
        background: #efe9df;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        letter-spacing: 0.01em;
    }

    .hero-kicker {
        color: #c45c26;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.2rem;
    }

    .hero-title {
        font-size: 2.05rem;
        font-weight: 750;
        line-height: 1.15;
        margin: 0 0 0.4rem 0;
        color: #1c1915;
    }

    .hero-copy {
        color: #5c564c;
        font-size: 1.05rem;
        max-width: 46rem;
        margin: 0 0 1.1rem 0;
    }

    .stepper {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin: 0.2rem 0 1.1rem 0;
    }

    .step {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.45rem 0.75rem;
        border-radius: 999px;
        background: #efe9df;
        color: #6b645a;
        font-size: 0.86rem;
        border: 1px solid #ddd4c6;
    }

    .step b {
        font-size: 0.72rem;
        width: 1.25rem;
        height: 1.25rem;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: #ddd4c6;
        color: #3f3a34;
    }

    .step.is-done {
        background: #e7f3ee;
        border-color: #b9d8cc;
        color: #215246;
    }

    .step.is-done b {
        background: #2a9d8f;
        color: white;
    }

    .step.is-current {
        background: #f7e4d6;
        border-color: #e2b08a;
        color: #7a3b14;
        font-weight: 650;
    }

    .step.is-current b {
        background: #c45c26;
        color: white;
    }

    .status-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 0.25rem 0 1rem 0;
    }

    .chip {
        background: white;
        border: 1px solid #ddd4c6;
        border-radius: 999px;
        padding: 0.35rem 0.7rem;
        font-size: 0.82rem;
        color: #3f3a34;
    }

    .chip strong {
        color: #1c1915;
    }

    .empty-card, .hint-card, .answer-card {
        background: white;
        border: 1px solid #ddd4c6;
        border-radius: 16px;
        padding: 1.15rem 1.25rem;
        box-shadow: 0 1px 0 rgba(28, 25, 21, 0.03);
    }

    .empty-card h3, .hint-card h3 {
        margin: 0 0 0.4rem 0;
    }

    .empty-card p, .hint-card p {
        color: #5c564c;
        margin: 0 0 0.75rem 0;
    }

    .empty-steps {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 0.75rem;
        margin: 1rem 0 0.25rem 0;
    }

    .empty-step {
        background: #f7f4ee;
        border-radius: 12px;
        padding: 0.85rem;
    }

    .empty-step span {
        display: block;
        color: #c45c26;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }

    .score-track {
        height: 8px;
        background: #efe9df;
        border-radius: 999px;
        overflow: hidden;
        margin: 0.35rem 0 0.15rem 0;
    }

    .score-fill {
        height: 100%;
        background: linear-gradient(90deg, #e07a3d, #c45c26);
        border-radius: 999px;
    }

    .prompt-block {
        background: #1c1915;
        color: #f4efe6;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.82rem;
        white-space: pre-wrap;
        overflow-x: auto;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #ddd4c6;
        border-radius: 14px;
        padding: 0.65rem 0.85rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.35rem;
    }

    .stTabs [data-baseweb="tab"] {
        background: #efe9df;
        border-radius: 10px 10px 0 0;
        padding: 0.55rem 0.9rem;
    }
</style>
"""


def inject_styles() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)
