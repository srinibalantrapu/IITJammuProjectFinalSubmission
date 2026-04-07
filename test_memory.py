"""Test session memory formatting."""

from __future__ import annotations

from memory import default_trip_inputs, format_conversation_context


def test_format_empty():
    assert format_conversation_context([]) == "(none)"
    assert format_conversation_context(None) == "(none)"


def test_format_messages():
    history = [
        {"role": "user", "content": "Plan Kyoto trip"},
        {"role": "assistant", "content": "Here is your plan..."},
    ]
    result = format_conversation_context(history)
    assert "USER:" in result
    assert "ASSISTANT:" in result


def test_format_truncates_long_assistant():
    long_content = "x" * 5000
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": long_content},
    ]
    result = format_conversation_context(history)
    assert "[...truncated...]" in result or len(result) < len(long_content)


def test_format_tuple_format():
    history = [["hi", "hello world"]]
    result = format_conversation_context(history)
    assert "hi" in result
    assert "hello world" in result


def test_default_trip_inputs_keys():
    inputs = default_trip_inputs()
    expected = {
        "user_message", "conversation_context", "destination",
        "dates_or_duration", "budget_hint", "interests", "travel_style",
        "prior_destination_json", "prior_budget_json", "prior_hotels_json",
        "stashed_prior_plan_json",
    }
    assert set(inputs.keys()) == expected
    assert inputs["conversation_context"] == "(none)"
