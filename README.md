# TruthRAG

### Self-Correcting RAG Pipeline with Semantic Chunking, Reranking & Evaluation

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/LangGraph-0.2-orange" alt="LangGraph">
  <img src="https://img.shields.io/badge/Ollama-local_LLM-black?logo=ollama" alt="Ollama">
  <img src="https://img.shields.io/badge/Qdrant-vector_DB-dc382c" alt="Qdrant">
  <img src="https://img.shields.io/badge/Streamlit-UI-ff4b4b?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Docker-containerized-2496ED?logo=docker&logoColor=white" alt="Docker">
</p>

**[Türkçe dokumantasyon](README.tr.md)**

---

Most RAG pipelines treat retrieval as a solved problem. They pull chunks, feed them to an LLM, and hope for the best. If the chunks are irrelevant or the model makes something up, nobody catches it until a user complains.

TruthRAG doesn't work that way. It scores every chunk before using it, checks the generated answer against sources, and regenerates if it finds inconsistencies. When local documents fall short, it searches the web as a fallback. Every answer comes with inline citations so you can verify claims yourself.

On top of that, it uses semantic chunking to split documents at natural topic boundaries, a cross-encoder to rerank results beyond what vector similarity alone can do, and query expansion to catch relevant content that a single phrasing might miss.

Everything runs locally -- no API keys, no cloud dependencies, your data stays on your machine.

---

## Pipeline

```
Question comes in
    |
    v
 1. QUERY EXPANSION -- LLM generates alternative phrasings to broaden recall
    |
    v
 2. HYBRID SEARCH -- vector similarity (70%) + BM25 keyword (30%)
    |                  combined via reciprocal rank fusion
    v
 3. CROSS-ENCODER RERANKING -- ms-marco-MiniLM rescores top candidates
    |
    v
 4. GRADING -- LLM scores each chunk: "How relevant is this?" (0.0-1.0)
    |
    +-- all low? --> WEB SEARCH (DuckDuckGo + Crawl4AI) --> re-grade
    |
    v
 5. GENERATE -- Answer with inline [Source N] citations
    |
    v
 6. HALLUCINATION CHECK -- Is the answer grounded in the sources?
    |
    +-- not grounded? --> regenerate (up to 2 retries)
    |
    v
 Result: answer + citations + confidence score + metadata
```

The flow is a **LangGraph StateGraph** -- each step is a node, routing decisions happen at conditional edges.

---

## Quick Start

**Requirements:** Docker and Docker Compose (v2+), at least 8 GB RAM.

```bash
git clone https://github.com/songulerdemguler/truthRAG.git
cd truthrag
cp .env.example .env
docker compose up -d --build
```

Pull the models on first run:

```bash
docker exec -it truthrag-ollama-1 ollama pull qwen3.5:2b
docker exec -it truthrag-ollama-1 ollama pull nomic-embed-text
```

Open **http://localhost:8501** in your browser. Upload a PDF or text file from the sidebar, ask a question, watch the answer stream in.

Each time the API starts, it clears the previous session's data -- only files you upload during the current session are used for retrieval.

---

## Key Features

**Semantic Chunking** -- Instead of splitting documents at fixed character counts (which often cuts sentences in half), TruthRAG uses embedding similarity to find natural topic boundaries. When the semantic distance between consecutive sentences exceeds a threshold, it starts a new chunk. This keeps related content together and produces more meaningful retrieval units.

**Cross-Encoder Reranking** -- Vector search is fast but approximate. After the initial hybrid retrieval, a cross-encoder model (`ms-marco-MiniLM-L-6-v2`) reads each query-chunk pair together and produces a more accurate relevance score. It's slower per pair but much better at distinguishing "actually relevant" from "superficially similar."

**Query Expansion** -- A single question can miss relevant content because of wording differences. The LLM generates 2 alternative phrasings, and all three queries are run against the index. Results are deduplicated so you get broader recall without redundancy.

**RAGAS Evaluation** -- Upload a test dataset (questions + expected answers) and get quantitative metrics: faithfulness (does the answer match the sources?), answer relevancy (does it address the question?), and context recall (did retrieval find the right content?). All computed locally using your own LLM.

**Chunk & Document Hit Rate** -- The analytics dashboard tracks which documents and chunks get used most often across queries. Useful for understanding what content is actually driving answers and what might be underutilized.

---

## API

Swagger docs: **http://localhost:8000/docs**

### Ask a question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?"}'
```

### Streaming (SSE)

```bash
curl -N -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?"}'
```

Tokens arrive one by one: first a metadata event with citations, then tokens, then a `done` event with latency.

### Chat sessions

```bash
# Create a session
curl -X POST http://localhost:8000/session
# {"session_id": "a1b2c3d4e5f67890"}

# Ask with context (remembers previous conversation)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Can you elaborate?", "session_id": "a1b2c3d4e5f67890"}'
```

### Upload documents

```bash
curl -X POST http://localhost:8000/ingest -F "file=@document.pdf"
```

50 MB limit. Accepts PDF, TXT, MD.

### Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /query` | Full pipeline query (30/min) |
| `POST /query/stream` | Streaming SSE query (30/min) |
| `POST /session` | Create a conversation session |
| `POST /ingest` | Upload and ingest a document (10/min) |
| `GET /analytics/summary?days=30` | Query count, avg confidence, latency, hallucinations |
| `GET /analytics/trend?days=30` | Confidence score over time |
| `GET /analytics/recent?limit=20` | Recent queries with metrics |
| `GET /analytics/chunk-hits?days=30` | Most frequently retrieved chunks |
| `GET /analytics/document-hits?days=30` | Most frequently retrieved documents |
| `GET /analytics/eval-summary?days=30` | RAGAS evaluation averages |
| `GET /analytics/eval-recent?limit=20` | Recent evaluation results |
| `POST /evaluate` | Run batch RAGAS evaluation (5/min) |
| `GET /health` | Qdrant + Ollama connectivity (200 or 503) |

Every response includes an `X-Correlation-ID` header for tracing.

---

## Configuration

All settings live in `.env`:

| Variable | Default | What it does |
|----------|---------|--------------|
| `LLM_MODEL` | `qwen3.5:2b` | Language model (must be pulled in Ollama) |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Ollama address |
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant address |
| `QDRANT_COLLECTION` | `truthrag` | Vector collection name |
| `TOP_K` | `5` | Chunks retrieved per query |
| `GRADE_THRESHOLD` | `0.5` | Minimum relevance score to keep a chunk |
| `MAX_RETRIES` | `2` | Regeneration attempts on hallucination |
| `BM25_WEIGHT` | `0.3` | Keyword search weight in hybrid fusion |
| `VECTOR_WEIGHT` | `0.7` | Vector search weight in hybrid fusion |
| `CHUNKING_STRATEGY` | `semantic` | `semantic` or `fixed` |
| `SEMANTIC_CHUNK_THRESHOLD` | `95.0` | Percentile breakpoint for semantic splits |
| `FIXED_CHUNK_SIZE` | `500` | Characters per chunk (when using fixed) |
| `FIXED_CHUNK_OVERLAP` | `50` | Overlap between fixed chunks |
| `RERANKER_ENABLED` | `true` | Enable cross-encoder reranking |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model |
| `RERANKER_TOP_K` | `10` | Candidates to rerank |
| `QUERY_EXPANSION_ENABLED` | `true` | Enable multi-query expansion |
| `QUERY_EXPANSION_COUNT` | `2` | Number of alternative queries |
| `RAGAS_EVAL_ENABLED` | `true` | Enable RAGAS evaluation endpoints |
| `MAX_CONVERSATION_TURNS` | `10` | History depth per session |
| `WEB_SEARCH_ENABLED` | `true` | Web fallback when local docs fail |
| `WEB_SEARCH_MAX_PAGES` | `3` | Pages to crawl per search |
| `WEB_SEARCH_TIMEOUT` | `15` | Per-page crawl timeout (seconds) |
| `PDF_PARSER` | `pymupdf` | `pymupdf` (fast) or `docling` (tables/layout) |

Running without Docker? Set `OLLAMA_BASE_URL=http://localhost:11434` and `QDRANT_URL=http://localhost:6333`.

---

## Tech Stack

| Component | Tool | Why |
|-----------|------|-----|
| LLM | Ollama + qwen3.5:2b | Fast local inference, no API keys |
| PDF parsing | PyMuPDF (default) / Docling (opt-in) | PyMuPDF is instant; Docling handles tables and complex layouts |
| Embeddings | nomic-embed-text | Lightweight, good quality for its size |
| Vector DB | Qdrant | Fast cosine similarity search with payload filtering |
| Keyword search | rank-bm25 | BM25Okapi for lexical matching |
| Reranking | sentence-transformers (cross-encoder) | Pairwise relevance scoring for precision |
| Chunking | LangChain SemanticChunker | Splits at topic boundaries, not arbitrary positions |
| Pipeline | LangGraph StateGraph | Multi-step orchestration with conditional routing |
| Web search | DuckDuckGo + Crawl4AI | No API keys, full page content extraction |
| Evaluation | RAGAS | Faithfulness, relevancy, recall metrics |
| API | FastAPI | REST + SSE streaming, rate limiting, CORS |
| UI | Streamlit | Chat interface, analytics dashboard, eval panel |
| Analytics | SQLite | Query metrics, chunk usage tracking |
| Infrastructure | Docker Compose | Everything in one `docker compose up` |

---

## Development

```bash
pip install -e ".[dev]"
pre-commit install

make lint          # ruff check
make type-check    # mypy
make test          # pytest
make check         # all at once
make format        # auto-fix + format
```

Running locally (without full Docker):

```bash
# Start only the infrastructure
docker run -d -p 6333:6333 qdrant/qdrant
docker run -d -p 11434:11434 -v ollama_data:/root/.ollama ollama/ollama
ollama pull qwen3.5:2b && ollama pull nomic-embed-text

# Run API and UI natively
uvicorn src.api.main:app --reload --port 8000
streamlit run ui/app.py  # separate terminal
```

---

## Project Structure

```
truthrag/
├── docker-compose.yml
├── Dockerfile / Dockerfile.ui
├── pyproject.toml
├── Makefile
│
├── src/
│   ├── config.py                  # All settings from .env
│   ├── utils.py                   # Singleton clients, JSON parsing, timers
│   ├── analytics.py               # SQLite: query log, eval log, chunk hits
│   ├── conversation.py            # In-memory session store
│   ├── ingestion/
│   │   ├── loader.py              # PyMuPDF / Docling (PDF), TextLoader (TXT/MD)
│   │   └── embedder.py            # Semantic or fixed chunking + Qdrant upsert
│   ├── retrieval/
│   │   ├── retriever.py           # Vector + BM25 + RRF hybrid search
│   │   └── reranker.py            # Cross-encoder reranking
│   ├── agents/
│   │   ├── grader.py              # Chunk relevance scoring
│   │   ├── generator.py           # Cited answer generation
│   │   ├── hallucination_checker.py
│   │   ├── query_expander.py      # Multi-query expansion
│   │   └── web_search.py          # DuckDuckGo + Crawl4AI fallback
│   ├── evaluation/
│   │   └── ragas_eval.py          # RAGAS batch + single evaluation
│   └── pipeline/
│       └── graph.py               # LangGraph StateGraph definition
│
├── ui/
│   └── app.py                     # Streamlit: chat, analytics, RAGAS eval
├── tests/
└── data/
    ├── ingest/                    # Upload your documents here
    └── eval/                      # RAGAS evaluation datasets
```
