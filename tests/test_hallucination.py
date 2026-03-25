"""Tests for src/agents/hallucination_checker.py."""

from unittest.mock import MagicMock

from src.agents.hallucination_checker import check_hallucination


class TestCheckHallucination:
    def test_no_hallucination(self, mock_llm, sample_chunks):
        mock_llm.invoke.return_value = MagicMock(
            content='{"hallucination": false, "confidence": 0.9, "issues": ""}'
        )
        result = check_hallucination("RAG uses retrieval.", sample_chunks, mock_llm)
        assert result["hallucination"] is False
        assert result["confidence"] == 0.9
        assert result["issues"] == ""

    def test_hallucination_detected(self, mock_llm, sample_chunks):
        mock_llm.invoke.return_value = MagicMock(
            content=(
                '{"hallucination": true, "confidence": 0.3, '
                '"issues": "answer claims unsupported facts"}'
            )
        )
        result = check_hallucination("Unsupported claim.", sample_chunks, mock_llm)
        assert result["hallucination"] is True
        assert result["confidence"] == 0.3

    def test_malformed_response_returns_safe_default(self, mock_llm, sample_chunks):
        mock_llm.invoke.return_value = MagicMock(content="Cannot determine.")
        result = check_hallucination("answer", sample_chunks, mock_llm)
        assert result["hallucination"] is False
        assert result["confidence"] == 0.5

    def test_llm_exception_returns_safe_default(self, mock_llm, sample_chunks):
        mock_llm.invoke.side_effect = RuntimeError("boom")
        result = check_hallucination("answer", sample_chunks, mock_llm)
        assert result["hallucination"] is False
