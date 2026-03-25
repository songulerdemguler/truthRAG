"""In-memory conversation store for multi-turn chat.

TODO: migrate to Redis or similar if we need persistence across restarts
"""

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field

from src.config import MAX_CONVERSATION_TURNS

logger = logging.getLogger(__name__)


@dataclass
class Turn:
    question: str
    answer: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class Session:
    session_id: str
    turns: list[Turn] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)


class ConversationStore:
    """Thread-safe in-memory conversation store with TTL expiry."""

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds

    def create_session(self) -> str:
        """Create a new conversation session and return its ID."""
        session_id = uuid.uuid4().hex[:16]
        with self._lock:
            self._sessions[session_id] = Session(session_id=session_id)
        logger.info("Created conversation session: %s", session_id)
        return session_id

    def add_turn(self, session_id: str, question: str, answer: str) -> None:
        """Record a Q&A turn in the session."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                session = Session(session_id=session_id)
                self._sessions[session_id] = session

            session.turns.append(Turn(question=question, answer=answer))
            session.last_active = time.time()

            # Trim to max turns
            if len(session.turns) > MAX_CONVERSATION_TURNS:
                session.turns = session.turns[-MAX_CONVERSATION_TURNS:]

    def get_history(self, session_id: str) -> str:
        """Format conversation history as a string for the LLM prompt."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session or not session.turns:
                return ""

        parts: list[str] = []
        for turn in session.turns:
            parts.append(f"User: {turn.question}")
            parts.append(f"Assistant: {turn.answer}")
        return "\n".join(parts)

    def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID, or None if expired/missing."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session and (time.time() - session.last_active) > self._ttl:
                del self._sessions[session_id]
                return None
            return session

    def cleanup_expired(self) -> int:
        """Remove expired sessions. Returns number of sessions removed."""
        now = time.time()
        removed = 0
        with self._lock:
            expired = [
                sid for sid, s in self._sessions.items() if (now - s.last_active) > self._ttl
            ]
            for sid in expired:
                del self._sessions[sid]
                removed += 1
        if removed:
            logger.info("Cleaned up %d expired sessions", removed)
        return removed


# module-level singleton
conversation_store = ConversationStore()
