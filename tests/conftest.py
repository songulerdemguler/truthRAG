"""Shared fixtures for TruthRAG tests."""

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_llm():
    """Return a mock ChatOllama that returns configurable content."""
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content='{"score": 0.8, "reason": "relevant"}')
    return llm


@pytest.fixture
def sample_chunks():
    """Return a list of sample retrieval chunks."""
    return [
        {
            "text": "RAG combines retrieval with generation for better answers.",
            "score": 0.9,
            "metadata": {"filename": "test.pdf", "chunk_index": 0, "page_number": 1},
        },
        {
            "text": "Vector databases store embeddings for semantic search.",
            "score": 0.85,
            "metadata": {"filename": "test.pdf", "chunk_index": 1, "page_number": 2},
        },
        {
            "text": "LLMs can generate fluent text from context.",
            "score": 0.7,
            "metadata": {"filename": "test.pdf", "chunk_index": 2, "page_number": 3},
        },
    ]


@pytest.fixture
def graded_chunks(sample_chunks):
    """Return chunks with grades attached."""
    grades = [0.9, 0.7, 0.3]
    return [
        {**c, "grade": g, "grade_reason": "test"}
        for c, g in zip(sample_chunks, grades, strict=False)
    ]
