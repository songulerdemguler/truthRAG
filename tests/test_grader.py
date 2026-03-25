"""Tests for src/agents/grader.py."""

from unittest.mock import MagicMock

from src.agents.grader import grade_chunk, grade_chunks


class TestGradeChunk:
    def test_valid_response(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(
            content='{"score": 0.85, "reason": "directly relevant"}'
        )
        result = grade_chunk("What is RAG?", "RAG is retrieval augmented generation.", mock_llm)
        assert result["score"] == 0.85
        assert result["reason"] == "directly relevant"

    def test_malformed_json_returns_zero(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(content="I cannot determine relevance")
        result = grade_chunk("question", "chunk", mock_llm)
        assert result["score"] == 0.0

    def test_llm_exception_returns_zero(self, mock_llm):
        mock_llm.invoke.side_effect = RuntimeError("connection timeout")
        result = grade_chunk("question", "chunk", mock_llm)
        assert result["score"] == 0.0

    def test_partial_json(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(content='{"score": 0.6}')
        result = grade_chunk("question", "chunk", mock_llm)
        assert result["score"] == 0.6
        assert result["reason"] == ""


class TestGradeChunks:
    def test_grades_all_chunks(self, mock_llm, sample_chunks):
        mock_llm.invoke.return_value = MagicMock(content='{"score": 0.7, "reason": "ok"}')
        results = grade_chunks("question", sample_chunks, mock_llm)
        assert len(results) == 3
        assert all("grade" in c for c in results)
        assert all(c["grade"] == 0.7 for c in results)

    def test_preserves_original_chunk_data(self, mock_llm, sample_chunks):
        mock_llm.invoke.return_value = MagicMock(content='{"score": 0.5, "reason": ""}')
        results = grade_chunks("question", sample_chunks, mock_llm)
        for original, graded in zip(sample_chunks, results, strict=False):
            assert graded["text"] == original["text"]
            assert graded["metadata"] == original["metadata"]
