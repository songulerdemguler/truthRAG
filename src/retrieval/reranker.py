"""Cross-encoder reranking for improved retrieval accuracy."""

import logging
from functools import lru_cache

from src.config import RERANKER_MODEL
from src.utils import StageTimer

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_cross_encoder():
    """Lazy-load the cross-encoder model (singleton)."""
    from sentence_transformers import CrossEncoder

    logger.info("Loading cross-encoder model: %s", RERANKER_MODEL)
    model = CrossEncoder(RERANKER_MODEL)
    return model


def rerank(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """Rerank chunks using a cross-encoder model.

    Scores each (query, chunk) pair and returns the top_k by cross-encoder score.
    """
    if not chunks:
        return chunks

    with StageTimer("cross_encoder_rerank"):
        model = _get_cross_encoder()
        pairs = [(query, c["text"]) for c in chunks]
        scores = model.predict(pairs)

        for chunk, score in zip(chunks, scores, strict=False):
            chunk["cross_encoder_score"] = float(score)

        ranked = sorted(chunks, key=lambda c: c["cross_encoder_score"], reverse=True)
        result = ranked[:top_k]

        logger.info(
            "Reranked %d → %d chunks (top score=%.3f)",
            len(chunks),
            len(result),
            result[0]["cross_encoder_score"] if result else 0,
        )
        return result
