"""SQLite-backed analytics for tracking query metrics."""

import logging
import sqlite3
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from src.config import ANALYTICS_DB

logger = logging.getLogger(__name__)

_lock = threading.Lock()


def _get_connection() -> sqlite3.Connection:
    """Create a new SQLite connection with WAL mode."""
    ANALYTICS_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(ANALYTICS_DB), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


@contextmanager
def _db() -> Generator[sqlite3.Connection, None, None]:
    """Thread-safe database connection context manager."""
    conn = _get_connection()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create analytics tables if they don't exist."""
    with _lock, _db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                session_id TEXT DEFAULT '',
                question TEXT NOT NULL,
                answer_length INTEGER DEFAULT 0,
                confidence_score REAL DEFAULT 0.0,
                hallucination_detected INTEGER DEFAULT 0,
                used_web_search INTEGER DEFAULT 0,
                retry_count INTEGER DEFAULT 0,
                num_sources INTEGER DEFAULT 0,
                latency_ms REAL DEFAULT 0.0
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_query_log_timestamp
            ON query_log (timestamp)
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS eval_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                question TEXT NOT NULL,
                faithfulness REAL DEFAULT 0.0,
                answer_relevancy REAL DEFAULT 0.0,
                context_recall REAL,
                eval_type TEXT DEFAULT 'single',
                batch_id TEXT DEFAULT ''
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_eval_log_timestamp
            ON eval_log (timestamp)
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunk_hits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                filename TEXT NOT NULL,
                chunk_index INTEGER DEFAULT 0,
                page_number INTEGER DEFAULT 0,
                score REAL DEFAULT 0.0
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunk_hits_filename
            ON chunk_hits (filename)
        """)
    logger.info("Analytics DB initialized at %s", ANALYTICS_DB)


def log_query(
    question: str,
    answer: str,
    confidence_score: float,
    hallucination_detected: bool,
    used_web_search: bool,
    retry_count: int,
    num_sources: int,
    latency_ms: float,
    session_id: str = "",
) -> None:
    """Record a query and its results."""
    with _lock, _db() as conn:
        conn.execute(
            """INSERT INTO query_log
               (timestamp, session_id, question, answer_length, confidence_score,
                hallucination_detected, used_web_search, retry_count, num_sources, latency_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                time.time(),
                session_id,
                question,
                len(answer),
                confidence_score,
                int(hallucination_detected),
                int(used_web_search),
                retry_count,
                num_sources,
                latency_ms,
            ),
        )


def log_chunk_hits(sources: list[dict]) -> None:
    """Record which chunks were used in a query response."""
    if not sources:
        return
    now = time.time()
    with _lock, _db() as conn:
        for s in sources:
            meta = s.get("metadata", {})
            conn.execute(
                "INSERT INTO chunk_hits (timestamp, filename, chunk_index, page_number, score) VALUES (?, ?, ?, ?, ?)",
                (
                    now,
                    meta.get("filename", s.get("filename", "unknown")),
                    meta.get("chunk_index", s.get("chunk_index", 0)),
                    meta.get("page_number", s.get("page_number", 0)),
                    float(s.get("score", 0.0)),
                ),
            )


def get_chunk_hit_rate(days: int = 30, limit: int = 20) -> list[dict]:
    """Get most frequently used chunks/documents."""
    cutoff = time.time() - (days * 86400)
    with _db() as conn:
        rows = conn.execute(
            """SELECT filename, chunk_index, page_number,
                      COUNT(*) as hit_count, AVG(score) as avg_score
               FROM chunk_hits WHERE timestamp > ?
               GROUP BY filename, chunk_index
               ORDER BY hit_count DESC LIMIT ?""",
            (cutoff, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def get_document_hit_rate(days: int = 30) -> list[dict]:
    """Get most frequently used documents (aggregated by filename)."""
    cutoff = time.time() - (days * 86400)
    with _db() as conn:
        rows = conn.execute(
            """SELECT filename, COUNT(*) as hit_count, AVG(score) as avg_score
               FROM chunk_hits WHERE timestamp > ?
               GROUP BY filename
               ORDER BY hit_count DESC""",
            (cutoff,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_summary(days: int = 30) -> dict[str, Any]:
    """Get aggregated stats for the last N days."""
    cutoff = time.time() - (days * 86400)
    with _db() as conn:
        row = conn.execute(
            """SELECT
                COUNT(*) as total_queries,
                AVG(confidence_score) as avg_confidence,
                AVG(latency_ms) as avg_latency_ms,
                SUM(hallucination_detected) as hallucination_count,
                SUM(used_web_search) as web_search_count,
                AVG(retry_count) as avg_retries,
                AVG(num_sources) as avg_sources
               FROM query_log WHERE timestamp > ?""",
            (cutoff,),
        ).fetchone()

        return {
            "total_queries": row["total_queries"] or 0,
            "avg_confidence": round(row["avg_confidence"] or 0, 3),
            "avg_latency_ms": round(row["avg_latency_ms"] or 0, 1),
            "hallucination_count": row["hallucination_count"] or 0,
            "web_search_count": row["web_search_count"] or 0,
            "avg_retries": round(row["avg_retries"] or 0, 2),
            "avg_sources": round(row["avg_sources"] or 0, 1),
            "days": days,
        }


def get_confidence_trend(days: int = 30, buckets: int = 20) -> list[dict]:
    """Get confidence score trend grouped into time buckets."""
    cutoff = time.time() - (days * 86400)
    bucket_size = (days * 86400) / buckets  # even time slices
    with _db() as conn:
        rows = conn.execute(
            """SELECT
                CAST((timestamp - ?) / ? AS INTEGER) as bucket,
                AVG(confidence_score) as avg_conf,
                COUNT(*) as count
               FROM query_log WHERE timestamp > ?
               GROUP BY bucket ORDER BY bucket""",
            (cutoff, bucket_size, cutoff),
        ).fetchall()

        return [
            {
                "bucket": r["bucket"],
                "avg_confidence": round(r["avg_conf"], 3),
                "query_count": r["count"],
            }
            for r in rows
        ]


def get_recent_queries(limit: int = 50) -> list[dict]:
    """Get the most recent queries with their metrics."""
    with _db() as conn:
        rows = conn.execute(
            """SELECT timestamp, question, confidence_score, hallucination_detected,
                      used_web_search, latency_ms, num_sources
               FROM query_log ORDER BY timestamp DESC LIMIT ?""",
            (limit,),
        ).fetchall()

        return [dict(r) for r in rows]


def log_evaluation(
    question: str,
    faithfulness: float,
    answer_relevancy: float,
    context_recall: float | None = None,
    eval_type: str = "single",
    batch_id: str = "",
) -> None:
    """Record a RAGAS evaluation result."""
    with _lock, _db() as conn:
        conn.execute(
            """INSERT INTO eval_log
               (timestamp, question, faithfulness, answer_relevancy, context_recall, eval_type, batch_id)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (time.time(), question, faithfulness, answer_relevancy, context_recall, eval_type, batch_id),
        )


def get_eval_summary(days: int = 30) -> dict[str, Any]:
    """Get aggregated RAGAS evaluation stats."""
    cutoff = time.time() - (days * 86400)
    with _db() as conn:
        row = conn.execute(
            """SELECT
                COUNT(*) as total_evals,
                AVG(faithfulness) as avg_faithfulness,
                AVG(answer_relevancy) as avg_answer_relevancy,
                AVG(context_recall) as avg_context_recall
               FROM eval_log WHERE timestamp > ?""",
            (cutoff,),
        ).fetchone()

        return {
            "total_evals": row["total_evals"] or 0,
            "avg_faithfulness": round(row["avg_faithfulness"] or 0, 3),
            "avg_answer_relevancy": round(row["avg_answer_relevancy"] or 0, 3),
            "avg_context_recall": round(row["avg_context_recall"] or 0, 3) if row["avg_context_recall"] else None,
            "days": days,
        }


def get_eval_recent(limit: int = 50) -> list[dict]:
    """Get the most recent RAGAS evaluation results."""
    with _db() as conn:
        rows = conn.execute(
            """SELECT timestamp, question, faithfulness, answer_relevancy,
                      context_recall, eval_type, batch_id
               FROM eval_log ORDER BY timestamp DESC LIMIT ?""",
            (limit,),
        ).fetchall()

        return [dict(r) for r in rows]
