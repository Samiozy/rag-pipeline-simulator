import pandas as pd
import streamlit as st

from rag_strategies.agentic.multi_agent.message import AgentMessage


def render_agent_messages(messages: list[AgentMessage]) -> None:
    if not messages:
        st.info("Run a query to see coordinator delegation and agent replies.")
        return
    rows = [
        {
            "from": message.sender,
            "to": message.recipient,
            "type": message.message_type,
            "content": message.content,
        }
        for message in messages
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    for message in messages:
        st.markdown(f"**{message.sender} → {message.recipient}** · {message.message_type}")
        st.caption(message.content)
