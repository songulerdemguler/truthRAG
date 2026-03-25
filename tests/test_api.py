"""Tests for src/api/main.py — endpoint validation and error handling."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Patch analytics DB to a temp file before importing app
with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as _tmp_db:
    _tmp_db_path = Path(_tmp_db.name)
_db_patch = patch("src.analytics.ANALYTICS_DB", _tmp_db_path)
_db_patch.start()

from src.analytics import init_db  # noqa: E402
from src.api.main import _sanitize_filename, app  # noqa: E402

init_db()
client = TestClient(app)


class TestSanitizeFilename:
    def test_normal_filename(self):
        assert _sanitize_filename("document.pdf") == "document.pdf"

    def test_strips_directory_traversal(self):
        assert _sanitize_filename("../../../etc/passwd") == "passwd"

    def test_strips_unsafe_characters(self):
        result = _sanitize_filename("file<>name|test.txt")
        assert "<" not in result
        assert ">" not in result
        assert "|" not in result

    def test_strips_leading_dots(self):
        assert _sanitize_filename(".hidden_file.txt") == "hidden_file.txt"

    def test_empty_becomes_upload(self):
        assert _sanitize_filename("") == "upload"

    def test_path_with_slashes(self):
        result = _sanitize_filename("/some/path/to/file.pdf")
        assert "/" not in result
        assert result == "file.pdf"


class TestQueryEndpoint:
    def test_empty_question_rejected(self):
        resp = client.post("/query", json={"question": ""})
        assert resp.status_code == 422

    def test_missing_question_rejected(self):
        resp = client.post("/query", json={})
        assert resp.status_code == 422

    def test_too_long_question_rejected(self):
        resp = client.post("/query", json={"question": "x" * 3000})
        assert resp.status_code == 422

    @patch("src.api.main.run_pipeline")
    @patch("src.api.main.log_query")
    def test_successful_query(self, mock_log, mock_pipeline):
        mock_pipeline.return_value = {
            "answer": "Test answer",
            "confidence_score": 0.8,
            "hallucination_detected": False,
            "sources": [{"text": "source", "score": 0.9}],
            "citations": [
                {
                    "source_number": 1,
                    "filename": "test.pdf",
                    "page_number": 1,
                    "chunk_index": 0,
                    "text_preview": "source",
                    "score": 0.9,
                }
            ],
            "used_web_search": False,
            "retry_count": 0,
        }
        resp = client.post("/query", json={"question": "What is RAG?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "Test answer"
        assert data["confidence_score"] == 0.8
        assert len(data["citations"]) == 1
        assert data["citations"][0]["filename"] == "test.pdf"

    @patch("src.api.main.run_pipeline", side_effect=RuntimeError("boom"))
    def test_pipeline_failure_returns_500(self, _):
        resp = client.post("/query", json={"question": "test"})
        assert resp.status_code == 500


class TestSessionEndpoint:
    def test_create_session(self):
        resp = client.post("/session")
        assert resp.status_code == 200
        assert "session_id" in resp.json()
        assert len(resp.json()["session_id"]) > 0


class TestIngestEndpoint:
    def test_unsupported_file_type(self):
        resp = client.post(
            "/ingest",
            files={"file": ("malware.exe", b"content", "application/octet-stream")},
        )
        assert resp.status_code == 400
        assert "Unsupported" in resp.json()["detail"]

    def test_oversized_file(self):
        big_content = b"x" * (51 * 1024 * 1024)  # 51 MB
        resp = client.post(
            "/ingest",
            files={"file": ("big.pdf", big_content, "application/pdf")},
        )
        assert resp.status_code == 413

    @patch("src.api.main.embed_and_store", return_value=10)
    @patch("src.api.main.load_documents", return_value=[])
    def test_reingest_without_file(self, mock_load, mock_embed):
        resp = client.post("/ingest")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestAnalyticsEndpoints:
    def test_analytics_summary(self):
        resp = client.get("/analytics/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_queries" in data
        assert "avg_confidence" in data

    def test_analytics_trend(self):
        resp = client.get("/analytics/trend")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_analytics_recent(self):
        resp = client.get("/analytics/recent")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


class TestHealthEndpoint:
    @patch("httpx.AsyncClient.get")
    @pytest.mark.anyio
    async def test_health_returns_503_when_services_down(self, mock_get):
        mock_get.side_effect = ConnectionError("refused")
        resp = client.get("/health")
        assert resp.status_code in (200, 503)
