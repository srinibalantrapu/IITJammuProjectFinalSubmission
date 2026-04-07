"""Test shared utility functions."""

from __future__ import annotations

from utils import content_to_str, dump_model, normalize_chat_history


def test_dump_model_none():
    assert dump_model(None) == {}


def test_dump_model_raw_fallback():
    assert dump_model(42) == {"raw": "42"}


class _FakeModel:
    def model_dump(self) -> dict:
        return {"key": "value"}


def test_dump_model_pydantic_like():
    assert dump_model(_FakeModel()) == {"key": "value"}


def test_content_to_str_none():
    assert content_to_str(None) == ""


def test_content_to_str_string():
    assert content_to_str("hello") == "hello"


def test_content_to_str_list():
    parts = [{"text": "a"}, "b", {"image": "x"}]
    assert content_to_str(parts) == "a\nb"


def test_content_to_str_other():
    assert content_to_str(123) == "123"


def test_normalize_empty():
    assert normalize_chat_history(None) == []
    assert normalize_chat_history([]) == []


def test_normalize_messages_format():
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    result = normalize_chat_history(msgs)
    assert len(result) == 2
    assert result[0]["role"] == "user"


def test_normalize_tuple_format():
    tuples = [["hi", "hello"], ["follow-up", "response"]]
    result = normalize_chat_history(tuples)
    assert len(result) == 4
    assert result[0] == {"role": "user", "content": "hi"}
    assert result[1] == {"role": "assistant", "content": "hello"}
