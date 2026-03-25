"""Tests for src/agents/generator.py."""

from unittest.mock import MagicMock

from src.agents.generator import _format_context, generate_answer


class TestFormatContext:
    def test_formats_numbered_sources(self, sample_chunks):
        context, citations = _format_context(sample_chunks)
        assert "[Source 1]" in context
        assert "[Source 2]" in context
        assert "test.pdf" in context
        assert len(citations) == 3

    def test_citations_contain_metadata(self, sample_chunks):
        _, citations = _format_context(sample_chunks)
        assert citations[0]["source_number"] == 1
        assert citations[0]["filename"] == "test.pdf"
        assert citations[0]["page_number"] == 1

    def test_empty_chunks(self):
        context, citations = _format_context([])
        assert context == ""
        assert citations == []


class TestGenerateAnswer:
    def test_returns_answer_and_citations(self, mock_llm, sample_chunks):
        mock_llm.invoke.return_value = MagicMock(content="RAG is a technique [Source 1].")
        answer, citations = generate_answer("What is RAG?", sample_chunks, mock_llm)
        assert answer == "RAG is a technique [Source 1]."
        assert len(citations) == 3

    def test_prompt_includes_context(self, mock_llm, sample_chunks):
        mock_llm.invoke.return_value = MagicMock(content="answer")
        generate_answer("What is RAG?", sample_chunks, mock_llm)

        call_args = mock_llm.invoke.call_args[0][0]
        assert "RAG combines retrieval" in call_args
        assert "What is RAG?" in call_args

    def test_prompt_includes_citation_instructions(self, mock_llm, sample_chunks):
        mock_llm.invoke.return_value = MagicMock(content="answer")
        generate_answer("What is RAG?", sample_chunks, mock_llm)

        call_args = mock_llm.invoke.call_args[0][0]
        assert "[Source N]" in call_args

    def test_llm_failure_returns_error(self, mock_llm, sample_chunks):
        mock_llm.invoke.side_effect = RuntimeError("timeout")
        answer, citations = generate_answer("question", sample_chunks, mock_llm)
        assert "error" in answer.lower()
        assert citations == []

    def test_empty_chunks_returns_answer(self, mock_llm):
        mock_llm.invoke.return_value = MagicMock(content="No context available.")
        answer, citations = generate_answer("question", [], mock_llm)
        assert isinstance(answer, str)
        assert len(answer) > 0
        assert citations == []

    def test_with_conversation_history(self, mock_llm, sample_chunks):
        mock_llm.invoke.return_value = MagicMock(content="Follow-up answer.")
        history = "User: What is RAG?\nAssistant: RAG is retrieval-augmented generation."
        answer, _ = generate_answer("Tell me more", sample_chunks, mock_llm, history)
        call_args = mock_llm.invoke.call_args[0][0]
        assert "Previous conversation" in call_args
        assert "What is RAG?" in call_args
