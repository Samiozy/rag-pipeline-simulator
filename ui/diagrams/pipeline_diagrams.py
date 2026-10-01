DIAGRAMS = {
    "naive": """
flowchart LR
    D[Documents] --> C[Chunks]
    C --> E[Embeddings]
    E --> V[Vector store]
    Q[Query] --> S[Top-K search]
    V --> S
    S --> P[Prompt]
    P --> G[Generator]
    G --> A[Answer]
""",
    "rerank": """
flowchart LR
    Q[Query] --> V[Vector search]
    V --> N[Top-N candidates]
    N --> R[Reranker]
    R --> K[Top-K]
    K --> P[Prompt]
    P --> G[Generator]
""",
    "hybrid": """
flowchart LR
    Q[Query] --> D[Dense retrieval]
    Q --> S[BM25]
    D --> F[Fusion]
    S --> F
    F --> K[Final Top-K]
    K --> P[Prompt]
    P --> G[Generator]
""",
    "graph": """
flowchart LR
    D[Documents] --> C[Chunks]
    C --> E[Entities]
    E --> G[Knowledge graph]
    Q[Query] --> M[Match entities]
    G --> M
    M --> T[Traverse]
    T --> K[Associated chunks]
    K --> P[Prompt]
""",
    "multimodal": """
flowchart LR
    D[PDF / files] --> T[Text]
    D --> I[Images]
    T --> TE[Text embeddings]
    I --> IE[Image captions]
    TE --> X[Shared index]
    IE --> X
    Q[Query] --> X
    X --> R[Mixed hits]
""",
    "agentic_router": """
flowchart LR
    Q[Query] --> R[Router]
    R --> RAG[RAG]
    R --> MEM[Memory]
    R --> TOOL[Tool]
    R --> DIR[Direct]
    RAG --> C[Context]
    MEM --> C
    TOOL --> C
    C --> G[Generator]
""",
    "multi_agent": """
flowchart LR
    Q[Query] --> CO[Coordinator]
    CO --> DOC[Document agent]
    CO --> SR[Search agent]
    CO --> GR[Graph agent]
    DOC --> SY[Synthesis]
    SR --> SY
    GR --> SY
    SY --> A[Answer]
""",
}


EDUCATION = {
    "naive": "Naive RAG retrieves the nearest chunks by vector similarity and sends them to the generator. It is the baseline everything else is compared with.",
    "rerank": "Retrieve-and-Rerank first pulls a wider candidate set, then a second relevance model reorders those chunks. The educational question: did the reranker change what the generator sees?",
    "hybrid": "Hybrid RAG combines keyword (BM25) and semantic (dense) retrieval, then fuses ranks. The educational question: which chunks appear only in one channel, and what does fusion keep?",
    "graph": "Graph RAG extracts entities and follows relationships instead of only vector similarity. The educational question: which neighboring entities pulled extra chunks into context?",
    "multimodal": "Multimodal RAG keeps text and image items in one index and labels each hit by modality. Images start as captions so the lab still runs offline.",
    "agentic_router": "A router chooses RAG, memory, a tool, or direct generation. You see the structured decision, not hidden chain-of-thought.",
    "multi_agent": "A coordinator delegates document, lexical, and graph retrieval, then a synthesis step merges the evidence.",
}


def mermaid(strategy_key: str) -> str:
    return DIAGRAMS.get(strategy_key, DIAGRAMS["naive"]).strip()
