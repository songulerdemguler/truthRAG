"""Splits documents into chunks, embeds them, and upserts into Qdrant."""

import logging
from functools import lru_cache

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from src.config import (
    CHUNKING_STRATEGY,
    FIXED_CHUNK_OVERLAP,
    FIXED_CHUNK_SIZE,
    QDRANT_COLLECTION,
    SEMANTIC_CHUNK_THRESHOLD,
)
from src.utils import get_embeddings, get_qdrant

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_splitter():
    """Return a text splitter based on the configured strategy."""
    if CHUNKING_STRATEGY == "semantic":
        try:
            from langchain_experimental.text_splitter import SemanticChunker

            embeddings = get_embeddings()
            splitter = SemanticChunker(
                embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=SEMANTIC_CHUNK_THRESHOLD,
            )
            logger.info("Using semantic chunking (threshold=%.1f)", SEMANTIC_CHUNK_THRESHOLD)
            return splitter
        except Exception:
            logger.warning("Semantic chunking failed, falling back to fixed-size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=FIXED_CHUNK_SIZE, chunk_overlap=FIXED_CHUNK_OVERLAP
    )
    logger.info("Using fixed-size chunking (%d/%d)", FIXED_CHUNK_SIZE, FIXED_CHUNK_OVERLAP)
    return splitter


def _ensure_collection(vector_size: int) -> None:
    """Create the Qdrant collection if it does not exist."""
    client = get_qdrant()
    collections = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in collections:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection: %s", QDRANT_COLLECTION)


def _already_ingested(filename: str) -> bool:
    """Check whether a file has already been ingested."""
    client = get_qdrant()
    try:
        result = client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="filename", match=MatchValue(value=filename))]
            ),
            limit=1,
        )
        return len(result[0]) > 0
    except Exception:
        logger.debug("Collection may not exist yet, treating as not ingested: %s", filename)
        return False


def embed_and_store(documents: list[Document]) -> int:
    """Chunk, embed, and store documents in Qdrant. Returns number of chunks added."""
    if not documents:
        return 0

    embeddings = get_embeddings()
    client = get_qdrant()
    splitter = _get_splitter()

    # Group documents by filename and skip already-ingested files
    by_file: dict[str, list[Document]] = {}
    for doc in documents:
        fname = doc.metadata.get("filename", "unknown")
        by_file.setdefault(fname, []).append(doc)

    new_docs: list[Document] = []
    for fname, docs in by_file.items():
        if _already_ingested(fname):
            logger.info("Skipping already-ingested file: %s", fname)
            continue
        new_docs.extend(docs)

    if not new_docs:
        logger.info("No new documents to ingest.")
        return 0

    chunks = splitter.split_documents(new_docs)
    texts = [chunk.page_content for chunk in chunks]
    vectors = embeddings.embed_documents(texts)

    _ensure_collection(len(vectors[0]))

    # Get current max id
    try:
        collection_info = client.get_collection(QDRANT_COLLECTION)
        start_id = collection_info.points_count or 0
    except Exception:
        start_id = 0

    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors, strict=False)):
        metadata = {
            "text": chunk.page_content,
            "filename": chunk.metadata.get("filename", "unknown"),
            "chunk_index": i,
            "page_number": chunk.metadata.get("page", 0),
        }
        points.append(PointStruct(id=start_id + i, vector=vector, payload=metadata))

    client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    logger.info("Stored %d chunks in Qdrant (strategy=%s).", len(points), CHUNKING_STRATEGY)
    return len(points)
