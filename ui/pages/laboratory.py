import streamlit as st

from core.prompts import DEFAULT_SYSTEM_PROMPT
from rag_strategies import strategy_labels
from ui.chat_view import render_transcript, selected_turn
from ui.chunk_view import render_chunks
from ui.common.agent_trace import render_agent_messages
from ui.common.fusion_table import render_fusion_table
from ui.common.graph_viewer import render_graph
from ui.common.router_panel import render_router_decision
from ui.common.trace_viewer import render_traces
from ui.components import SAMPLE_PATH, render_example_questions
from ui.diagrams.pipeline_diagrams import EDUCATION, mermaid
from ui.document_view import render_documents
from ui.metrics_view import render_metrics
from ui.prompt_view import render_prompt
from ui.retrieval_view import render_retrieval


def render_laboratory(strategy_key: str, documents, chunks, strategy, config) -> None:
    st.markdown(f"### {strategy_labels().get(strategy_key, strategy.label)}")
    st.write(EDUCATION.get(strategy_key, strategy.description))
    st.code(mermaid(strategy_key), language=None)

    conversation = st.session_state.get("conversation") or []
    using_sample = any(document.source == SAMPLE_PATH.name for document in documents)

    tabs = ["Documents", "Chunks", "Retrieval", "Context", "Prompt", "Answer", "Trace", "Metrics"]
    if strategy_key == "rerank":
        tabs.insert(3, "Reranking")
    if strategy_key == "hybrid":
        tabs[2:3] = ["Dense", "Sparse", "Fusion"]
    if strategy_key == "graph":
        tabs.insert(2, "Graph")
    if strategy_key == "multimodal":
        tabs.insert(3, "Modalities")
    if strategy_key == "agentic_router":
        tabs.insert(2, "Router")
    if strategy_key == "multi_agent":
        tabs.insert(2, "Agents")

    panels = st.tabs(tabs)
    mapping = {name: panels[index] for index, name in enumerate(tabs)}

    with mapping["Documents"]:
        render_documents(documents)
    with mapping["Chunks"]:
        render_chunks(chunks)

    output = conversation[-1] if conversation else None
    extras = output.extras if output else {}

    if "Retrieval" in mapping:
        with mapping["Retrieval"]:
            if output:
                render_retrieval(output.retrieved)
            else:
                st.info("Run a query to inspect retrieved chunks.")
    if "Dense" in mapping:
        with mapping["Dense"]:
            render_retrieval(extras.get("dense") or [])
        with mapping["Sparse"]:
            render_retrieval(extras.get("sparse") or [])
        with mapping["Fusion"]:
            render_fusion_table(extras.get("fusion_rows") or [])
    if "Graph" in mapping:
        with mapping["Graph"]:
            store = getattr(strategy, "graph_store", None)
            if store is not None:
                render_graph(store, extras)
            else:
                st.info("Graph store is not attached to this strategy.")
    if "Modalities" in mapping:
        with mapping["Modalities"]:
            labels = extras.get("modalities") or []
            if labels:
                for index, label in enumerate(labels, 1):
                    st.write(f"#{index} {label.capitalize()}")
            else:
                st.info("Run a query to see text vs image hit labels.")
    if "Router" in mapping:
        with mapping["Router"]:
            decision = extras.get("decision")
            if decision:
                render_router_decision(decision, extras)
            else:
                st.info("Run a query to see the structured route decision.")
    if "Agents" in mapping:
        with mapping["Agents"]:
            render_agent_messages(extras.get("messages") or [])
    if "Reranking" in mapping:
        with mapping["Reranking"]:
            st.caption("Left: first-pass candidates. Right: reranked Top-K sent to the generator.")
            left, right = st.columns(2)
            with left:
                st.markdown("**Candidates**")
                render_retrieval(extras.get("candidates") or [])
            with right:
                st.markdown("**After rerank**")
                render_retrieval(extras.get("reranked") or (output.retrieved if output else []))
    if "Context" in mapping:
        with mapping["Context"]:
            if output and output.context:
                for item in output.context:
                    with st.expander(f"{item.source_id} · {item.score if item.score is not None else ''}"):
                        st.write(item.content)
            else:
                st.info("Context items appear after a run.")
    with mapping["Prompt"]:
        render_prompt(selected_turn(conversation, f"prompt_{strategy_key}") if conversation else None)
    with mapping["Answer"]:
        render_transcript(conversation)
    with mapping["Trace"]:
        render_traces(output.traces if output else [])
    with mapping["Metrics"]:
        render_metrics(output, conversation)

    st.divider()
    st.markdown("#### Run")
    if using_sample and not conversation:
        render_example_questions()
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
    with st.expander("Instructions for the generator", expanded=False):
        st.text_area("System / RAG instruction", height=120, key="system_prompt", label_visibility="collapsed")

    st.caption("Follow-ups stay on this strategy and corpus. Use New conversation or New document session in the header.")

    incoming = st.session_state.get("pending_question")
    chat_prompt = st.chat_input("Ask a question, or follow up on the last answer…")
    if chat_prompt:
        incoming = chat_prompt.strip()
    if incoming:
        try:
            with st.spinner("Retrieving and generating…"):
                history = [(turn.question, turn.answer) for turn in conversation]
                result = strategy.run(
                    incoming,
                    history=history,
                    temperature=config["temperature"],
                    max_tokens=config["max_tokens"],
                    system_prompt=st.session_state.system_prompt,
                )
                if result.extras.get("retrieval_query"):
                    result.retrieval_query = result.extras["retrieval_query"]
                conversation.append(result)
                st.session_state["conversation"] = conversation
                st.session_state["last_output"] = result
                st.session_state.pop("pending_question", None)
            st.rerun()
        except Exception as exc:
            st.error(f"The query failed: {exc}")
