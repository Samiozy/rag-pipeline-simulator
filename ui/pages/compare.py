import streamlit as st

from core.evaluation import retrieval_summary
from core.prompts import DEFAULT_SYSTEM_PROMPT
from core.services.compare import compare_runs
from core.services.lab import build_strategy
from rag_strategies import strategy_labels
from ui.retrieval_view import render_retrieval


def render_compare(documents, chunks, components, config) -> None:
    st.markdown("### Compare two RAG configurations")
    st.write(
        "The same question is run against two independent strategies that share this corpus. "
        "The laboratory reports differences. It does not declare a winner."
    )
    labels = strategy_labels()
    key_a = config["compare_a"]
    key_b = config["compare_b"]
    st.caption(f"Pipeline A: **{labels[key_a]}** · Pipeline B: **{labels[key_b]}**")

    question = st.text_area("Question", placeholder="Ask something the documents can answer…", key="compare_question")
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
    if st.button("Run comparison", type="primary", disabled=not question.strip()):
        try:
            with st.spinner("Running both pipelines…"):
                common = dict(
                    retrieval=config["retrieval"],
                    rerank=config["rerank"],
                    hybrid=config["hybrid"],
                    graph=config.get("graph"),
                    multimodal=config.get("multimodal"),
                    router=config.get("router"),
                    multi_agent=config.get("multi_agent"),
                )
                strategy_a = build_strategy(key_a, components, **common)
                strategy_b = build_strategy(key_b, components, **common)
                strategy_a.chunks = chunks
                strategy_b.chunks = chunks
                for strategy in (strategy_a, strategy_b):
                    inner = getattr(strategy, "document_strategy", None)
                    if inner is not None:
                        inner.chunks = chunks
                experiment = compare_runs(
                    question.strip(),
                    strategy_a,
                    strategy_b,
                    config_a={"strategy": key_a},
                    config_b={"strategy": key_b},
                    temperature=config["temperature"],
                    max_tokens=config["max_tokens"],
                    system_prompt=st.session_state.system_prompt,
                )
                st.session_state["compare_result"] = experiment
        except Exception as exc:
            st.error(f"Comparison failed: {exc}")

    experiment = st.session_state.get("compare_result")
    if not experiment:
        st.info("Run a comparison to see overlap, latency, retrieved chunks, and answers.")
        return

    overlap = experiment.overlap
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("A ∩ B", overlap.get("both", 0))
    m2.metric("A only", overlap.get("a_only", 0))
    m3.metric("B only", overlap.get("b_only", 0))
    m4.metric("A total latency", f"{experiment.pipeline_a.total_ms:.0f} ms")
    m5.metric("B total latency", f"{experiment.pipeline_b.total_ms:.0f} ms")

    summary_a = retrieval_summary(experiment.pipeline_a.retrieved)
    summary_b = retrieval_summary(experiment.pipeline_b.retrieved)
    st.dataframe(
        {
            "Metric": [
                "Retrieval latency (ms)",
                "Generation latency (ms)",
                "Total latency (ms)",
                "Retrieved chunks",
                "Context characters",
                "Context tokens (est.)",
                "Average score",
            ],
            labels[key_a]: [
                f"{experiment.pipeline_a.retrieval_ms:.1f}",
                f"{experiment.pipeline_a.generation_ms:.1f}",
                f"{experiment.pipeline_a.total_ms:.1f}",
                summary_a["retrieved"],
                experiment.pipeline_a.context_chars,
                experiment.pipeline_a.context_tokens_est,
                f"{summary_a['avg_score']:.3f}",
            ],
            labels[key_b]: [
                f"{experiment.pipeline_b.retrieval_ms:.1f}",
                f"{experiment.pipeline_b.generation_ms:.1f}",
                f"{experiment.pipeline_b.total_ms:.1f}",
                summary_b["retrieved"],
                experiment.pipeline_b.context_chars,
                experiment.pipeline_b.context_tokens_est,
                f"{summary_b['avg_score']:.3f}",
            ],
        },
        use_container_width=True,
        hide_index=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader(labels[key_a])
        st.write(experiment.pipeline_a.answer)
        render_retrieval(experiment.pipeline_a.retrieved)
    with col_b:
        st.subheader(labels[key_b])
        st.write(experiment.pipeline_b.answer)
        render_retrieval(experiment.pipeline_b.retrieved)
