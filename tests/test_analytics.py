"""Tests for src/analytics.py."""

import tempfile
from pathlib import Path
from unittest.mock import patch

from src.analytics import get_confidence_trend, get_recent_queries, get_summary, init_db, log_query


class TestAnalytics:
    def setup_method(self):
        """Use a temp DB for each test."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            self._db_path = Path(tmp.name)
        self._patcher = patch("src.analytics.ANALYTICS_DB", self._db_path)
        self._patcher.start()
        init_db()

    def teardown_method(self):
        self._patcher.stop()
        self._db_path.unlink(missing_ok=True)

    def test_init_db_creates_table(self):
        summary = get_summary()
        assert summary["total_queries"] == 0

    def test_log_and_retrieve(self):
        log_query(
            question="What is RAG?",
            answer="RAG is retrieval-augmented generation.",
            confidence_score=0.85,
            hallucination_detected=False,
            used_web_search=False,
            retry_count=0,
            num_sources=3,
            latency_ms=1200.5,
            session_id="test-session",
        )
        summary = get_summary()
        assert summary["total_queries"] == 1
        assert summary["avg_confidence"] == 0.85

    def test_recent_queries(self):
        log_query("q1", "a1", 0.7, False, False, 0, 2, 500.0)
        log_query("q2", "a2", 0.9, False, True, 1, 4, 800.0)
        recent = get_recent_queries(limit=10)
        assert len(recent) == 2
        assert recent[0]["question"] == "q2"  # most recent first

    def test_confidence_trend(self):
        for i in range(5):
            log_query(f"q{i}", f"a{i}", 0.5 + i * 0.1, False, False, 0, 2, 500.0)
        trend = get_confidence_trend(days=30, buckets=5)
        assert isinstance(trend, list)

    def test_summary_with_no_data(self):
        summary = get_summary(days=1)
        assert summary["total_queries"] == 0
        assert summary["avg_confidence"] == 0
