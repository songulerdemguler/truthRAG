"""Tests for src/utils.py — JSON parsing, correlation IDs, timer."""

from src.utils import StageTimer, get_correlation_id, new_correlation_id, parse_llm_json


class TestParseLlmJson:
    def test_valid_json(self):
        result = parse_llm_json('{"score": 0.8, "reason": "good"}')
        assert result["score"] == 0.8
        assert result["reason"] == "good"

    def test_json_with_surrounding_text(self):
        result = parse_llm_json('Here is the result: {"score": 0.5} hope that helps!')
        assert result["score"] == 0.5

    def test_json_in_markdown_fence(self):
        text = '```json\n{"score": 0.9, "reason": "very relevant"}\n```'
        result = parse_llm_json(text)
        assert result["score"] == 0.9

    def test_empty_string_returns_default(self):
        result = parse_llm_json("", default={"score": 0.0})
        assert result == {"score": 0.0}

    def test_no_json_returns_default(self):
        result = parse_llm_json("This is just plain text with no JSON", default={"x": 1})
        assert result == {"x": 1}

    def test_malformed_json_returns_default(self):
        result = parse_llm_json('{"score": }', default={"score": 0.0})
        assert result == {"score": 0.0}

    def test_none_default_gives_empty_dict(self):
        result = parse_llm_json("no json here")
        assert result == {}

    def test_nested_json(self):
        text = '{"outer": {"inner": 1}, "value": true}'
        result = parse_llm_json(text)
        assert result["outer"]["inner"] == 1
        assert result["value"] is True

    def test_json_with_booleans(self):
        result = parse_llm_json('{"hallucination": false, "confidence": 0.7}')
        assert result["hallucination"] is False
        assert result["confidence"] == 0.7


class TestCorrelationId:
    def test_new_id_is_set_and_retrievable(self):
        cid = new_correlation_id()
        assert len(cid) == 12
        assert get_correlation_id() == cid

    def test_new_id_replaces_previous(self):
        first = new_correlation_id()
        second = new_correlation_id()
        assert first != second
        assert get_correlation_id() == second


class TestStageTimer:
    def test_timer_records_elapsed(self):
        with StageTimer("test_stage") as timer:
            _ = sum(range(1000))
        assert timer.elapsed > 0
        assert timer.stage_name == "test_stage"
