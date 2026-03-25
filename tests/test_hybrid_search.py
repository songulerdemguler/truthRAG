"""Tests for src/retrieval/retriever.py — hybrid search components."""

from src.retrieval.retriever import _bm25_rerank, _reciprocal_rank_fusion, _tokenize


class TestTokenize:
    def test_basic_tokenization(self):
        tokens = _tokenize("Hello World! This is a test.")
        assert tokens == ["hello", "world", "this", "is", "a", "test"]

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_special_characters(self):
        tokens = _tokenize("RAG-based system (v2.0)")
        assert "rag" in tokens
        assert "based" in tokens


class TestBm25Rerank:
    def test_adds_bm25_scores(self):
        # BM25 needs 3+ docs for meaningful IDF scores
        chunks = [
            {"text": "retrieval augmented generation system combines search", "score": 0.9},
            {"text": "weather forecast tomorrow sunny cloudy rain", "score": 0.8},
            {"text": "machine learning neural network deep learning", "score": 0.7},
        ]
        result = _bm25_rerank("retrieval generation", chunks)
        assert "bm25_score" in result[0]
        assert "bm25_score" in result[1]
        # First chunk should score higher for "retrieval generation"
        assert result[0]["bm25_score"] > result[1]["bm25_score"]

    def test_empty_chunks(self):
        assert _bm25_rerank("query", []) == []

    def test_scores_normalized_to_0_1(self):
        chunks = [
            {"text": "RAG system", "score": 0.9},
            {"text": "another doc", "score": 0.8},
        ]
        result = _bm25_rerank("RAG", chunks)
        for c in result:
            assert 0.0 <= c["bm25_score"] <= 1.0


class TestReciprocalRankFusion:
    def test_combines_scores(self):
        chunks = [
            {"text": "a", "score": 0.9, "bm25_score": 0.5},
            {"text": "b", "score": 0.5, "bm25_score": 0.9},
        ]
        result = _reciprocal_rank_fusion(chunks)
        assert all("hybrid_score" in c for c in result)

    def test_ordering(self):
        chunks = [
            {"text": "a", "score": 0.9, "bm25_score": 0.9},
            {"text": "b", "score": 0.1, "bm25_score": 0.1},
        ]
        result = _reciprocal_rank_fusion(chunks)
        assert result[0]["text"] == "a"
        assert result[0]["hybrid_score"] > result[1]["hybrid_score"]
