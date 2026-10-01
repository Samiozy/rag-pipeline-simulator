import pandas as pd
import streamlit as st

from rag_strategies.graph.graph_store import GraphStore


def render_graph(store: GraphStore, extras: dict) -> None:
    stats = store.stats()
    st.caption(f"{stats['nodes']} nodes · {stats['edges']} edges · backend {stats['backend']}")
    if not store.nodes:
        st.warning("No entities were extracted. Try a longer document.")
        return
    matched = extras.get("matched_nodes") or []
    visited = extras.get("visited_nodes") or list(store.nodes)
    st.write(f"Query entities: {', '.join(extras.get('query_entities') or []) or '—'}")
    st.write(f"Matched nodes: {', '.join(matched) or '—'}")
    st.write(f"Traversal depth: {extras.get('depth', '—')}")
    node_rows = [
        {"entity": node.name, "type": node.entity_type, "chunks": len(node.chunk_ids), "selected": node.name.lower() in {item.lower() for item in visited}}
        for node in store.nodes.values()
    ]
    st.dataframe(pd.DataFrame(node_rows), use_container_width=True, hide_index=True)
    edge_rows = [
        {"source": store.nodes.get(edge.source, edge).name if not isinstance(store.nodes.get(edge.source), str) else edge.source, "relation": edge.relation, "target": store.nodes[edge.target].name if edge.target in store.nodes else edge.target}
        for edge in store.subgraph_edges(visited) or store.edges[:40]
    ]
    if edge_rows:
        st.dataframe(pd.DataFrame(edge_rows), use_container_width=True, hide_index=True)
    lines = ["flowchart LR"]
    for edge in (store.subgraph_edges(visited) or store.edges)[:30]:
        src = edge.source.replace(" ", "_")
        dst = edge.target.replace(" ", "_")
        lines.append(f"    {src} -->|{edge.relation}| {dst}")
    if len(lines) > 1:
        st.code("\n".join(lines), language=None)
