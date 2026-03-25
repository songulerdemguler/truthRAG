"""Shared utilities - LLM/DB clients, JSON parsing, timing."""

import json
import logging
import time
import uuid
from contextvars import ContextVar
from functools import lru_cache
from typing import Any

from langchain_ollama import ChatOllama, OllamaEmbeddings
from qdrant_client import QdrantClient

from src.config import EMBED_MODEL, LLM_MODEL, OLLAMA_BASE_URL, QDRANT_URL

logger = logging.getLogger(__name__)

# -- Correlation ID for request tracing

_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")


def new_correlation_id() -> str:
    """Generate and set a new correlation ID for the current request."""
    cid = uuid.uuid4().hex[:12]
    _correlation_id.set(cid)
    return cid


def get_correlation_id() -> str:
    """Get the current request's correlation ID."""
    return _correlation_id.get()


# Singleton clients (cached so we don't re-init on every call)


@lru_cache(maxsize=1)
def get_llm() -> ChatOllama:
    """Return a singleton ChatOllama instance."""
    logger.info("Initializing ChatOllama (model=%s)", LLM_MODEL)
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    return llm.bind(think=False)


@lru_cache(maxsize=1)
def get_embeddings() -> OllamaEmbeddings:
    """Return a singleton OllamaEmbeddings instance."""
    logger.info("Initializing OllamaEmbeddings (model=%s)", EMBED_MODEL)
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)


@lru_cache(maxsize=1)
def get_qdrant() -> QdrantClient:
    """Return a singleton QdrantClient instance."""
    logger.info("Initializing QdrantClient (url=%s)", QDRANT_URL)
    return QdrantClient(url=QDRANT_URL)


# JSON parsing - LLMs love to wrap JSON in markdown fences etc.


def parse_llm_json(text: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
    """Extract and parse JSON from an LLM response string.

    Handles common LLM quirks: markdown fences, leading text, trailing text.
    Returns *default* (or empty dict) on any parse failure.
    """
    if default is None:
        default = {}

    if not text or not text.strip():
        return default

    content = text.strip()

    # Strip markdown code fences if present
    if "```" in content:
        parts = content.split("```")
        for part in parts:
            cleaned = part.strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            if cleaned.startswith("{"):
                content = cleaned
                break

    # Extract the outermost JSON object
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        logger.debug("No JSON object found in LLM response: %.100s", content)
        return default

    json_str = content[start : end + 1]

    try:
        result: dict[str, Any] = json.loads(json_str)
        return result
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse failed: %s | raw: %.200s", exc, json_str)
        return default


# Timing helper for pipeline stages


class StageTimer:
    """Context manager that logs elapsed time for a pipeline stage."""

    def __init__(self, stage_name: str) -> None:
        self.stage_name = stage_name
        self.start: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "StageTimer":
        self.start = time.perf_counter()
        cid = get_correlation_id()
        prefix = f"[{cid}] " if cid else ""
        logger.info("%sStage '%s' started", prefix, self.stage_name)
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed = time.perf_counter() - self.start
        cid = get_correlation_id()
        prefix = f"[{cid}] " if cid else ""
        logger.info("%sStage '%s' finished in %.2fs", prefix, self.stage_name, self.elapsed)
