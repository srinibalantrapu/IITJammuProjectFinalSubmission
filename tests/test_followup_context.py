"""Test JSON extraction and history-merge logic in followup_context."""

from __future__ import annotations

from followup_context import (
    _extract_json_block,
    _loads_embedded_json_object,
    _loads_travel_plan_dict,
    merge_trip_inputs_from_history,
    wrap_refinement_user_message,
)

SAMPLE_PLAN = '{"destination": {"destination_summary": "Kyoto"}, "itinerary": {"days": [1,2,3]}, "budget": {}}'


def test_loads_travel_plan_dict_valid():
    assert _loads_travel_plan_dict(SAMPLE_PLAN) is not None


def test_loads_travel_plan_dict_rejects_non_plan():
    assert _loads_travel_plan_dict('{"foo": "bar"}') is None
    assert _loads_travel_plan_dict("") is None
    assert _loads_travel_plan_dict("not json") is None


def test_extract_json_block_fenced():
    text = "Here is the result:\n```json\n" + SAMPLE_PLAN + "\n```\nDone."
    result = _extract_json_block(text)
    assert result is not None
    assert result["destination"]["destination_summary"] == "Kyoto"


def test_extract_json_block_raw():
    result = _extract_json_block(SAMPLE_PLAN)
    assert result is not None


def test_extract_json_block_embedded():
    text = "Some preamble " + SAMPLE_PLAN + " trailing text"
    result = _extract_json_block(text)
    assert result is not None


def test_extract_json_block_empty():
    assert _extract_json_block("") is None
    assert _extract_json_block(None) is None


def test_loads_embedded_json_object_no_brace():
    assert _loads_embedded_json_object("no json here") is None


def test_merge_trip_inputs_fills_destination():
    history = [
        {"role": "user", "content": "Plan a trip to Kyoto"},
        {"role": "assistant", "content": f"```json\n{SAMPLE_PLAN}\n```"},
    ]
    base = {
        "destination": "",
        "dates_or_duration": "",
        "budget_hint": "mid-range",
        "user_message": "",
        "conversation_context": "(none)",
        "interests": "",
        "travel_style": "",
        "stashed_prior_plan_json": "(none)",
    }
    merged = merge_trip_inputs_from_history(base, history)
    assert "Kyoto" in merged["destination"]
    assert "3 days" in merged["dates_or_duration"]


def test_merge_trip_inputs_preserves_existing():
    history = [
        {"role": "assistant", "content": f"```json\n{SAMPLE_PLAN}\n```"},
    ]
    base = {
        "destination": "Tokyo",
        "dates_or_duration": "7 days",
        "budget_hint": "",
        "user_message": "",
        "conversation_context": "(none)",
        "interests": "",
        "travel_style": "",
        "stashed_prior_plan_json": "(none)",
    }
    merged = merge_trip_inputs_from_history(base, history)
    assert merged["destination"] == "Tokyo"
    assert merged["dates_or_duration"] == "7 days"


def test_wrap_refinement_adds_prefix():
    result = wrap_refinement_user_message("Make it cheaper", "USER: Plan Kyoto\nASSISTANT: ...")
    assert result.startswith("[Refinement")
    assert "Make it cheaper" in result


def test_wrap_refinement_no_context():
    assert wrap_refinement_user_message("Hello", "(none)") == "Hello"
    assert wrap_refinement_user_message("Hello", "") == "Hello"
