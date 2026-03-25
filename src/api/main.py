"""FastAPI app - main entry point for the TruthRAG API."""

import asyncio
import json
import logging
import re
import shutil
import tempfile
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.responses import StreamingResponse

from src.analytics import (
    get_chunk_hit_rate,
    get_confidence_trend,
    get_document_hit_rate,
    get_eval_recent,
    get_eval_summary,
    get_recent_queries,
    log_chunk_hits,
    get_summary,
    init_db,
    log_evaluation,
    log_query,
)
from src.config import INGEST_DIR, OLLAMA_BASE_URL, QDRANT_URL, RAGAS_EVAL_ENABLED
from src.conversation import conversation_store
from src.ingestion.embedder import embed_and_store
from src.ingestion.loader import load_documents
from src.pipeline.graph import run_pipeline
from src.utils import get_correlation_id, get_llm, new_correlation_id

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# App setup


def _clean_previous_session() -> None:
    """Delete Qdrant collection and clear ingest directory for a fresh session."""
    from src.config import QDRANT_COLLECTION
    from src.utils import get_qdrant

    # Clear Qdrant collection
    try:
        client = get_qdrant()
        collections = [c.name for c in client.get_collections().collections]
        if QDRANT_COLLECTION in collections:
            client.delete_collection(QDRANT_COLLECTION)
            logger.info("Cleared Qdrant collection: %s", QDRANT_COLLECTION)
    except Exception:
        logger.warning("Could not clear Qdrant collection on startup.")

    # Clear ingest directory (keep .gitkeep)
    if INGEST_DIR.exists():
        for f in INGEST_DIR.iterdir():
            if f.name == ".gitkeep":
                continue
            try:
                f.unlink()
            except Exception:
                logger.warning("Could not remove %s", f)
        logger.info("Cleared ingest directory.")


@asynccontextmanager
async def lifespan(_application: FastAPI):
    """Initialize services on startup."""
    init_db()
    _clean_previous_session()
    yield


app = FastAPI(title="TruthRAG API", version="2.0.0", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Try again later."},
    )


# Correlation ID middleware
@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    cid = new_correlation_id()
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = cid
    return response


# Upload limits & filename sanitization

MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}
MAX_QUESTION_LENGTH = 2000

_SAFE_FILENAME_RE = re.compile(r"[^\w\s\-.]")


def _sanitize_filename(name: str) -> str:
    """Remove path traversal and unsafe characters from a filename."""
    name = Path(name).name
    name = _SAFE_FILENAME_RE.sub("_", name)
    name = name.lstrip(".")
    return name or "upload"


# Request / Response models


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_LENGTH)
    session_id: str = Field(default="", max_length=64)


class CitationItem(BaseModel):
    source_number: int
    filename: str
    page_number: int
    chunk_index: int
    text_preview: str
    score: float


class SourceItem(BaseModel):
    text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    confidence_score: float
    hallucination_detected: bool
    sources: list[SourceItem]
    citations: list[CitationItem]
    used_web_search: bool
    retry_count: int
    session_id: str


class IngestResponse(BaseModel):
    status: str
    chunks_added: int


class HealthResponse(BaseModel):
    status: str
    qdrant: bool
    ollama: bool


class SessionResponse(BaseModel):
    session_id: str


class AnalyticsSummaryResponse(BaseModel):
    total_queries: int
    avg_confidence: float
    avg_latency_ms: float
    hallucination_count: int
    web_search_count: int
    avg_retries: float
    avg_sources: float
    days: int


# Endpoints


@app.post("/session", response_model=SessionResponse)
async def create_session() -> SessionResponse:
    """Create a new conversation session."""
    session_id = conversation_store.create_session()
    return SessionResponse(session_id=session_id)


@app.post("/query", response_model=QueryResponse)
@limiter.limit("30/minute")
async def query(request: Request, req: QueryRequest) -> QueryResponse:
    """Run the full RAG pipeline for a question, with conversation context."""
    cid = get_correlation_id()
    logger.info("[%s] Query received: %s", cid, req.question[:100])

    # Get conversation history if session provided
    conversation_history = ""
    session_id = req.session_id
    if session_id:
        conversation_history = conversation_store.get_history(session_id)

    start = time.perf_counter()
    try:
        result = await asyncio.to_thread(
            run_pipeline, req.question, session_id, conversation_history
        )
    except Exception as exc:
        logger.exception("[%s] Pipeline execution failed", cid)
        raise HTTPException(status_code=500, detail="Pipeline execution failed") from exc

    latency_ms = (time.perf_counter() - start) * 1000

    # Store conversation turn
    if session_id:
        conversation_store.add_turn(session_id, req.question, result["answer"])

    # Log analytics
    log_query(
        question=req.question,
        answer=result["answer"],
        confidence_score=result["confidence_score"],
        hallucination_detected=result["hallucination_detected"],
        used_web_search=result["used_web_search"],
        retry_count=result["retry_count"],
        num_sources=len(result.get("sources", [])),
        latency_ms=latency_ms,
        session_id=session_id,
    )
    log_chunk_hits(result.get("citations", []))

    return QueryResponse(**result, session_id=session_id)


@app.post("/query/stream")
@limiter.limit("30/minute")
async def query_stream(request: Request, req: QueryRequest) -> StreamingResponse:
    """Stream the RAG pipeline answer token-by-token via SSE."""
    cid = get_correlation_id()
    logger.info("[%s] Stream query received: %s", cid, req.question[:100])

    # Get conversation history
    conversation_history = ""
    session_id = req.session_id
    if session_id:
        conversation_history = conversation_store.get_history(session_id)

    async def event_stream() -> AsyncGenerator[str, None]:
        start = time.perf_counter()
        try:
            # Run query expansion + retrieval + grading
            from src.agents.generator import _format_context
            from src.agents.grader import grade_chunks
            from src.agents.query_expander import expand_query
            from src.config import GRADE_THRESHOLD, QUERY_EXPANSION_ENABLED
            from src.retrieval.retriever import retrieve

            llm = get_llm()

            # Query expansion
            if QUERY_EXPANSION_ENABLED:
                queries = await asyncio.to_thread(expand_query, req.question, llm)
            else:
                queries = [req.question]

            # Multi-query retrieval
            all_chunks: list[dict] = []
            seen: set[str] = set()
            for q in queries:
                for chunk in await asyncio.to_thread(retrieve, q):
                    key = chunk["text"].strip()[:200]
                    if key not in seen:
                        seen.add(key)
                        all_chunks.append(chunk)
            chunks = all_chunks
            graded = await asyncio.to_thread(grade_chunks, req.question, chunks, llm)
            filtered = [c for c in graded if c["grade"] >= GRADE_THRESHOLD]
            chunks_to_use = filtered or graded

            # Send metadata event
            context, citations = _format_context(chunks_to_use)
            meta = {
                "type": "metadata",
                "citations": citations,
                "num_sources": len(chunks_to_use),
                "used_web_search": False,
            }
            yield f"data: {json.dumps(meta)}\n\n"

            # Stream the answer tokens
            from src.agents.generator import GENERATE_PROMPT, GENERATE_WITH_HISTORY_PROMPT

            if conversation_history:
                prompt = GENERATE_WITH_HISTORY_PROMPT.format(
                    history=conversation_history, context=context, question=req.question
                )
            else:
                prompt = GENERATE_PROMPT.format(context=context, question=req.question)

            full_answer = ""
            async for chunk in llm.astream(prompt):
                token = str(chunk.content)
                full_answer += token
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            latency_ms = (time.perf_counter() - start) * 1000

            # Store turn and log
            if session_id:
                conversation_store.add_turn(session_id, req.question, full_answer)

            log_query(
                question=req.question,
                answer=full_answer,
                confidence_score=sum(c["grade"] for c in graded) / len(graded) if graded else 0,
                hallucination_detected=False,
                used_web_search=False,
                retry_count=0,
                num_sources=len(chunks_to_use),
                latency_ms=latency_ms,
                session_id=session_id,
            )

            # Send done event
            yield f"data: {json.dumps({'type': 'done', 'latency_ms': round(latency_ms, 1)})}\n\n"

        except Exception as exc:
            logger.exception("[%s] Stream failed", cid)
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/ingest", response_model=IngestResponse)
@limiter.limit("10/minute")
async def ingest(request: Request, file: UploadFile | None = File(None)) -> IngestResponse:
    """Ingest a single uploaded file, or re-ingest all files in data/ingest/."""
    cid = get_correlation_id()

    if file and file.filename:
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
            )

        content = await file.read()
        if len(content) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {MAX_UPLOAD_SIZE // (1024 * 1024)} MB",
            )

        safe_name = _sanitize_filename(file.filename)
        if not any(safe_name.endswith(e) for e in ALLOWED_EXTENSIONS):
            safe_name += ext

        INGEST_DIR.mkdir(parents=True, exist_ok=True)
        dest = INGEST_DIR / safe_name
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        shutil.move(str(tmp_path), str(dest))
        logger.info("[%s] Saved uploaded file: %s (%d bytes)", cid, safe_name, len(content))

    try:
        documents = await asyncio.to_thread(load_documents)
        chunks_added = await asyncio.to_thread(embed_and_store, documents)
    except Exception as exc:
        logger.exception("[%s] Ingestion failed", cid)
        raise HTTPException(status_code=500, detail="Ingestion failed") from exc

    return IngestResponse(status="ok", chunks_added=chunks_added)


# Analytics


@app.get("/analytics/summary", response_model=AnalyticsSummaryResponse)
async def analytics_summary(days: int = 30) -> AnalyticsSummaryResponse:
    """Get aggregated query analytics."""
    data = await asyncio.to_thread(get_summary, days)
    return AnalyticsSummaryResponse(**data)


@app.get("/analytics/trend")
async def analytics_trend(days: int = 30) -> list[dict]:
    """Get confidence score trend over time."""
    return await asyncio.to_thread(get_confidence_trend, days)


@app.get("/analytics/recent")
async def analytics_recent(limit: int = 50) -> list[dict]:
    """Get recent queries and their metrics."""
    return await asyncio.to_thread(get_recent_queries, limit)


@app.get("/analytics/chunk-hits")
async def chunk_hits(days: int = 30, limit: int = 20) -> list[dict]:
    """Get most frequently used chunks."""
    return await asyncio.to_thread(get_chunk_hit_rate, days, limit)


@app.get("/analytics/document-hits")
async def document_hits(days: int = 30) -> list[dict]:
    """Get most frequently used documents."""
    return await asyncio.to_thread(get_document_hit_rate, days)


# RAGAS Evaluation


@app.get("/analytics/eval-summary")
async def eval_summary(days: int = 30) -> dict:
    """Get aggregated RAGAS evaluation metrics."""
    return await asyncio.to_thread(get_eval_summary, days)


@app.get("/analytics/eval-recent")
async def eval_recent(limit: int = 50) -> list[dict]:
    """Get recent RAGAS evaluation results."""
    return await asyncio.to_thread(get_eval_recent, limit)


@app.post("/evaluate")
@limiter.limit("5/minute")
async def run_evaluation(request: Request, file: UploadFile = File(...)) -> dict:
    """Upload a JSON evaluation dataset and run RAGAS batch evaluation."""
    import json as json_module
    import uuid

    from src.evaluation.ragas_eval import evaluate_batch, save_dataset

    content = await file.read()
    try:
        data = json_module.loads(content)
    except json_module.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file")

    if not isinstance(data, list) or not data:
        raise HTTPException(status_code=400, detail="Expected a JSON array of {question, ground_truth} objects")

    batch_id = str(uuid.uuid4())[:8]
    filename = f"eval_{batch_id}.json"
    dataset_path = await asyncio.to_thread(save_dataset, filename, data)

    results = await asyncio.to_thread(evaluate_batch, str(dataset_path))

    # Log results
    for r in results:
        log_evaluation(
            question=r.get("question", ""),
            faithfulness=r.get("faithfulness", 0.0),
            answer_relevancy=r.get("answer_relevancy", 0.0),
            context_recall=r.get("context_recall"),
            eval_type="batch",
            batch_id=batch_id,
        )

    return {"batch_id": batch_id, "total": len(results), "results": results}


# Health check


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse | JSONResponse:
    """Check connectivity to Qdrant and Ollama."""
    qdrant_ok = False
    ollama_ok = False

    async with httpx.AsyncClient(timeout=5) as client:
        try:
            resp = await client.get(f"{QDRANT_URL}/healthz")
            qdrant_ok = resp.status_code == 200
        except Exception:
            logger.warning("Qdrant health check failed")

        try:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            ollama_ok = resp.status_code == 200
        except Exception:
            logger.warning("Ollama health check failed")

    result = HealthResponse(status="ok", qdrant=qdrant_ok, ollama=ollama_ok)

    if not (qdrant_ok and ollama_ok):
        return JSONResponse(status_code=503, content=result.model_dump())

    return result
