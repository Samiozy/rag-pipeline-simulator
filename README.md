# RAG Pipeline Simulator

A modular, local-first Retrieval-Augmented Generation (RAG) laboratory built with Python and Streamlit. The application is designed to expose the internal stages of RAG rather than hide them behind a chatbot interface.

## Quick start with Ollama or another local LLM

If you want to run generation locally without external APIs, you can use Ollama or any OpenAI-compatible local server.

1. Install and start Ollama.
2. Pull a model, for example:
   ```bash
   ollama pull llama3.2
   ```
3. Start the model:
   ```bash
   ollama run llama3.2
   ```
4. In the app sidebar, choose `Ollama` as the Generator.
5. Keep the model name as `llama3.2` (or the model you pulled).
6. Leave the API key blank.
7. Keep the default base URL:
   ```text
   http://localhost:11434/v1
   ```

You can also select `Local LLM (OpenAI-compatible)` if your server uses a different OpenAI-compatible endpoint.

## What you can inspect

`Document → Cleaning → Chunking → Embedding → Vector Store → Retrieval → Reranking → Prompt → Generation → Metrics`

The simulator currently supports:

- PDF, TXT, Markdown, and DOCX ingestion
- Recursive-character, fixed-character, and sentence-aware chunking
- Sentence Transformers embeddings for fully local retrieval
- OpenAI embeddings as an optional alternative
- FAISS or pure NumPy cosine-similarity search
- Optional cross-encoder reranking
- Offline extractive generation
- Local LLM generation via Ollama and other OpenAI-compatible backends
- OpenAI and OpenAI-compatible generation
- Anthropic generation
- Google Gemini generation
- Retrieved-chunk inspection, scores, source/page metadata, exact prompt inspection, and latency metrics

## Why the architecture is LLM-agnostic

The RAG pipeline depends on abstract interfaces rather than a provider SDK. `rag/generation/base.py` defines the generator contract, while provider-specific adapters implement it. The pipeline itself only calls:

```python
answer = generator.generate(prompt)
```

You can therefore add another model provider without changing retrieval, chunking, indexing, or the Streamlit UI architecture.

The same pattern is used for `EmbeddingProvider`, `VectorStore`, `Chunker`, and `Reranker`.

## Project structure

```text
rag-pipeline-simulator/
├── app.py
├── config/
│   ├── __init__.py
│   └── settings.py
├── data/
│   └── sample.txt
├── models/
│   ├── __init__.py
│   ├── chunk.py
│   ├── document.py
│   └── retrieval_result.py
├── rag/
│   ├── pipeline.py
│   ├── ingestion/
│   │   ├── base.py
│   │   ├── factory.py
│   │   ├── pdf_loader.py
│   │   ├── text_loader.py
│   │   └── docx_loader.py
│   ├── preprocessing/
│   │   └── cleaner.py
│   ├── chunking/
│   │   ├── base.py
│   │   ├── fixed_chunker.py
│   │   ├── recursive_chunker.py
│   │   └── sentence_chunker.py
│   ├── embeddings/
│   │   ├── base.py
│   │   ├── factory.py
│   │   ├── sentence_transformer.py
│   │   └── openai_embedding.py
│   ├── vectorstores/
│   │   ├── base.py
│   │   ├── faiss_store.py
│   │   └── numpy_store.py
│   ├── retrieval/
│   │   └── reranker.py
│   ├── prompts/
│   │   └── rag_prompt.py
│   ├── generation/
│   │   ├── base.py
│   │   ├── factory.py
│   │   ├── extractive.py
│   │   ├── openai_compatible.py
│   │   ├── anthropic_generator.py
│   │   └── gemini_generator.py
│   └── evaluation/
│       └── retrieval_metrics.py
├── ui/
│   ├── sidebar.py
│   ├── document_view.py
│   ├── chunk_view.py
│   └── retrieval_view.py
├── tests/
│   ├── test_chunking.py
│   ├── test_prompt.py
│   └── test_vectorstore.py
├── .env.example
├── .gitignore
└── requirements.txt
```

## Local setup

### 1. Create a virtual environment

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Optional API configuration

The application works without an API key when you select:

- `Sentence Transformers` embeddings
- `Extractive (offline)` generator

To use hosted models:

```bash
cp .env.example .env
```

Then add only the keys you need.

### 4. Run the simulator

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

## Fastest first test

1. Run the app.
2. Upload `data/sample.txt`.
3. Keep `Sentence Transformers` as the embedding provider.
4. Keep `FAISS` as the vector store.
5. Keep `Extractive (offline)` as the generator.
6. Ask: `Why should retrieval be evaluated independently from generation?`
7. Inspect the `Retrieval & Answer`, `Prompt Inspector`, and `Metrics` tabs.

No API key is required for this path.

## Using Ollama, vLLM, LM Studio, or another OpenAI-compatible server

The sidebar includes a dedicated `Ollama` preset and a more general `Local LLM (OpenAI-compatible)` option.

For Ollama, the default base URL is already configured as:

```text
http://localhost:11434/v1
```

Typical setup:

1. Start your local model server, for example:
   ```bash
   ollama run llama3.2
   ```
2. In the app, choose `Ollama` under the Generator selector.
3. Enter the model name exactly as exposed by the server, such as `llama3.2`.
4. Leave the API key blank unless your local server requires authentication.
5. Keep the default base URL or change it to your server's `/v1` endpoint.

The more general `Local LLM (OpenAI-compatible)` option works for other servers that implement the OpenAI chat-completions interface, such as local proxies or LM Studio.

If the local server does not require authentication, the adapter supplies a placeholder key because the OpenAI client expects one syntactically.

## Adding another LLM provider

Create a new adapter:

```python
from rag.generation.base import Generator

class MyProviderGenerator(Generator):
    def __init__(self, model_name: str, ...):
        self.model_name = model_name

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 700) -> str:
        # Call the provider here.
        return "provider response"
```

Then register it in `rag/generation/factory.py` and add the provider name to `ui/sidebar.py`. No changes are needed in `rag/pipeline.py`.

## Design notes

The Streamlit layer is intentionally thin. `app.py` handles user interaction and session state, while the actual RAG behavior lives under `rag/`. This makes it possible to reuse the same pipeline later from FastAPI, a notebook, CLI, automated evaluation suite, or agent system.

The first release focuses on transparent single-run inspection. Good next extensions are side-by-side experiment comparison, retrieval relevance labels and Recall@K/MRR/NDCG, semantic chunking, hybrid BM25+dense retrieval, persistent vector databases, multiple corpora, query rewriting, multi-query retrieval, contextual compression, and RAGAS-style generation evaluation.

## Tests

Run:

```bash
pytest -q
```

The included tests cover chunk creation, prompt composition, and basic vector retrieval.

## Security

Do not commit `.env` or provider API keys. `.env` is already included in `.gitignore`.
