"""Hybrid retriever: combines vector similarity with BM25 via reciprocal rank fusion."""

import logging
import re

from rank_bm25 import BM25Okapi

from src.config import BM25_WEIGHT, QDRANT_COLLECTION, RERANKER_ENABLED, RERANKER_TOP_K, TOP_K, VECTOR_WEIGHT
from src.utils import StageTimer, get_embeddings, get_qdrant

logger = logging.getLogger(__name__)

_TOKENIZE_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return _TOKENIZE_RE.findall(text.lower())


def _vector_search(query_vector: list[float], limit: int) -> list[dict]:
    """Pure vector similarity search against Qdrant."""
    client = get_qdrant()
    results = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_vector,
        limit=limit,
    )
    chunks: list[dict] = []
    for hit in results:
        payload = hit.payload or {}
        chunks.append(
            {
                "text": payload.get("text", ""),
                "score": hit.score,
                "metadata": {
                    "filename": payload.get("filename", ""),
                    "chunk_index": payload.get("chunk_index", 0),
                    "page_number": payload.get("page_number", 0),
                },
            }
        )
    return chunks


def _bm25_rerank(query: str, chunks: list[dict]) -> list[dict]:
    """Score chunks with BM25 and attach bm25_score."""
    if not chunks:
        return chunks

    corpus = [_tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(corpus)
    query_tokens = _tokenize(query)
    scores = bm25.get_scores(query_tokens)

    # normalize to 0-1 range, avoid div by zero
    max_score = max(scores) if max(scores) > 0 else 1.0
    for chunk, score in zip(chunks, scores, strict=False):
        chunk["bm25_score"] = float(score / max_score)

    return chunks


def _reciprocal_rank_fusion(chunks: list[dict], k: int = 60) -> list[dict]:
    """Combine vector and BM25 scores using weighted reciprocal rank fusion."""
    # Sort by vector score (descending) to get vector ranks
    vector_sorted = sorted(enumerate(chunks), key=lambda x: x[1]["score"], reverse=True)
    vector_ranks = {idx: rank + 1 for rank, (idx, _) in enumerate(vector_sorted)}

    # Sort by BM25 score (descending) to get BM25 ranks
    bm25_sorted = sorted(enumerate(chunks), key=lambda x: x[1].get("bm25_score", 0), reverse=True)
    bm25_ranks = {idx: rank + 1 for rank, (idx, _) in enumerate(bm25_sorted)}

    # Compute RRF score
    for idx, chunk in enumerate(chunks):
        vector_rrf = VECTOR_WEIGHT / (k + vector_ranks[idx])
        bm25_rrf = BM25_WEIGHT / (k + bm25_ranks[idx])
        chunk["hybrid_score"] = vector_rrf + bm25_rrf

    return sorted(chunks, key=lambda c: c["hybrid_score"], reverse=True)


def retrieve(query: str, top_k: int | None = None) -> list[dict]:
    """Return top-k chunks using hybrid search (vector + BM25 + RRF).

    Each result is a dict with keys: text, score, metadata, bm25_score, hybrid_score.
    """
    top_k = top_k or TOP_K
    # grab more candidates than needed so BM25 reranking has room to work
    fetch_limit = min(top_k * 3, 30)

    with StageTimer("hybrid_retrieve"):
        try:
            embeddings = get_embeddings()
            query_vector = embeddings.embed_query(query)

            # Step 1: Vector search
            chunks = _vector_search(query_vector, limit=fetch_limit)
            if not chunks:
                return []

            # Step 2: BM25 rerank
            chunks = _bm25_rerank(query, chunks)

            # Step 3: Reciprocal rank fusion
            chunks = _reciprocal_rank_fusion(chunks)

            # Step 4: Cross-encoder reranking
            if RERANKER_ENABLED:
                from src.retrieval.reranker import rerank

                chunks = rerank(query, chunks[:RERANKER_TOP_K], top_k)

            # Return top-k
            result = chunks[:top_k]
            logger.info(
                "Hybrid search: %d candidates → %d results for query.",
                len(chunks),
                len(result),
            )
            return result
        except Exception:
            logger.exception("Hybrid retrieval failed")
            return []
