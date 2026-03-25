"""App configuration - all settings loaded from .env"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "truthrag")
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen3")
TOP_K: int = int(os.getenv("TOP_K", "5"))
GRADE_THRESHOLD: float = float(os.getenv("GRADE_THRESHOLD", "0.5"))
MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "2"))
INGEST_DIR: Path = BASE_DIR / "data" / "ingest"

# Hybrid search
BM25_WEIGHT: float = float(os.getenv("BM25_WEIGHT", "0.3"))
VECTOR_WEIGHT: float = float(os.getenv("VECTOR_WEIGHT", "0.7"))

# Chunking strategy
CHUNKING_STRATEGY: str = os.getenv("CHUNKING_STRATEGY", "semantic")  # "semantic" or "fixed"
SEMANTIC_CHUNK_THRESHOLD: float = float(os.getenv("SEMANTIC_CHUNK_THRESHOLD", "95.0"))
FIXED_CHUNK_SIZE: int = int(os.getenv("FIXED_CHUNK_SIZE", "500"))
FIXED_CHUNK_OVERLAP: int = int(os.getenv("FIXED_CHUNK_OVERLAP", "50"))

# Cross-encoder reranking
RERANKER_ENABLED: bool = os.getenv("RERANKER_ENABLED", "true").lower() == "true"
RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_TOP_K: int = int(os.getenv("RERANKER_TOP_K", "10"))

# Query expansion
QUERY_EXPANSION_ENABLED: bool = os.getenv("QUERY_EXPANSION_ENABLED", "true").lower() == "true"
QUERY_EXPANSION_COUNT: int = int(os.getenv("QUERY_EXPANSION_COUNT", "2"))

# Analytics
ANALYTICS_DB: Path = BASE_DIR / "data" / "analytics.db"

# Conversation memory
MAX_CONVERSATION_TURNS: int = int(os.getenv("MAX_CONVERSATION_TURNS", "10"))

# Web search (DuckDuckGo + Crawl4AI)
WEB_SEARCH_ENABLED: bool = os.getenv("WEB_SEARCH_ENABLED", "true").lower() == "true"
WEB_SEARCH_MAX_PAGES: int = int(os.getenv("WEB_SEARCH_MAX_PAGES", "3"))
WEB_SEARCH_CONTENT_LIMIT: int = int(os.getenv("WEB_SEARCH_CONTENT_LIMIT", "2000"))
WEB_SEARCH_TIMEOUT: int = int(os.getenv("WEB_SEARCH_TIMEOUT", "15"))

# RAGAS evaluation
RAGAS_EVAL_ENABLED: bool = os.getenv("RAGAS_EVAL_ENABLED", "true").lower() == "true"
EVAL_DATASET_DIR: Path = BASE_DIR / "data" / "eval"
