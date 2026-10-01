import pandas as pd
import streamlit as st

from core.fusion import FusionRow


def render_fusion_table(rows: list[FusionRow]) -> None:
    if not rows:
        st.info("Run a hybrid query to see how dense and sparse ranks were fused.")
        return
    frame = pd.DataFrame(
        [
            {
                "fused_rank": row.fused_rank,
                "fused_score": round(row.fused_score, 4),
                "dense_rank": row.dense_rank if row.dense_rank is not None else "—",
                "dense_score": round(row.dense_score, 4) if row.dense_score is not None else "—",
                "sparse_rank": row.sparse_rank if row.sparse_rank is not None else "—",
                "bm25_score": round(row.sparse_score, 4) if row.sparse_score is not None else "—",
                "source": row.source,
                "chunk": row.preview,
            }
            for row in rows
        ]
    )
    st.dataframe(frame, use_container_width=True, hide_index=True)
    st.caption("Changing fusion method or channel weights should change this table. The tool does not declare a winner.")
