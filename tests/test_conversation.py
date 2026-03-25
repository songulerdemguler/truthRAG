"""Tests for src/conversation.py."""

import time
from unittest.mock import patch

from src.conversation import ConversationStore


class TestConversationStore:
    def test_create_session(self):
        store = ConversationStore()
        sid = store.create_session()
        assert len(sid) == 16
        assert store.get_session(sid) is not None

    def test_add_turn_and_get_history(self):
        store = ConversationStore()
        sid = store.create_session()
        store.add_turn(sid, "What is RAG?", "RAG is retrieval-augmented generation.")
        store.add_turn(sid, "Tell me more", "It combines retrieval with LLMs.")

        history = store.get_history(sid)
        assert "What is RAG?" in history
        assert "RAG is retrieval-augmented generation." in history
        assert "Tell me more" in history

    def test_auto_creates_session_on_add(self):
        store = ConversationStore()
        store.add_turn("new-session", "q", "a")
        assert store.get_session("new-session") is not None

    def test_empty_history_for_missing_session(self):
        store = ConversationStore()
        assert store.get_history("nonexistent") == ""

    def test_max_turns_trimming(self):
        with patch("src.conversation.MAX_CONVERSATION_TURNS", 3):
            store = ConversationStore()
            sid = store.create_session()
            for i in range(5):
                store.add_turn(sid, f"q{i}", f"a{i}")
            session = store.get_session(sid)
            assert len(session.turns) == 3
            assert session.turns[0].question == "q2"  # oldest trimmed

    def test_expired_session_returns_none(self):
        store = ConversationStore(ttl_seconds=0)
        sid = store.create_session()
        time.sleep(0.01)
        assert store.get_session(sid) is None

    def test_cleanup_expired(self):
        store = ConversationStore(ttl_seconds=0)
        store.create_session()
        store.create_session()
        time.sleep(0.01)
        removed = store.cleanup_expired()
        assert removed == 2
