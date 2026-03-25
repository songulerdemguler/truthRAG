"""Tests for src/pipeline/graph.py — routing logic and state management."""

from src.pipeline.graph import route_after_grade, route_after_hallucination


class TestRouteAfterGrade:
    def test_routes_to_generate_when_filtered_chunks_exist(self):
        state = {"filtered_chunks": [{"text": "chunk", "grade": 0.8}]}
        assert route_after_grade(state) == "generate"

    def test_routes_to_web_search_when_no_filtered_chunks(self):
        state = {"filtered_chunks": []}
        assert route_after_grade(state) == "web_search"

    def test_routes_to_web_search_when_key_missing(self):
        state = {}
        assert route_after_grade(state) == "web_search"


class TestRouteAfterHallucination:
    def test_routes_to_done_when_no_hallucination(self):
        state = {"hallucination_detected": False, "retry_count": 0}
        assert route_after_hallucination(state) == "done"

    def test_routes_to_regenerate_when_hallucination_and_retries_left(self):
        state = {"hallucination_detected": True, "retry_count": 0}
        assert route_after_hallucination(state) == "regenerate"

    def test_routes_to_done_when_hallucination_but_max_retries(self):
        state = {"hallucination_detected": True, "retry_count": 2}
        assert route_after_hallucination(state) == "done"

    def test_routes_to_done_when_retries_exceed_max(self):
        state = {"hallucination_detected": True, "retry_count": 5}
        assert route_after_hallucination(state) == "done"
