"""Merge form slots from prior assistant JSON; tag refinement requests for the crew."""

from __future__ import annotations

import json
import re
from typing import Any


def _loads_travel_plan_dict(raw: str) -> dict[str, Any] | None:
    """Parse JSON; accept only objects that look like our pipeline report."""
    raw = raw.strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    # Avoid treating unrelated JSON as a prior plan
    if not any(k in obj for k in ("destination", "itinerary", "budget", "hotels")):
        return None
    return obj


def _loads_embedded_json_object(text: str) -> dict[str, Any] | None:
    """If the model prefixed prose, parse the first top-level `{...}` via raw_decode."""
    start = text.find("{")
    if start == -1:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text, start)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    if not any(k in obj for k in ("destination", "itinerary", "budget", "hotels")):
        return None
    return obj


def _extract_json_block(text: str) -> dict[str, Any] | None:
    """
    Recover the last assistant travel plan from message content.

    Order: (1) ```json ... ``` fences (Gradio default), (2) whole-string JSON,
    (3) first embedded JSON object (prose before/after).
    """
    if not text:
        return None
    stripped = text.strip()

    if "```" in stripped:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", stripped)
        if m:
            parsed = _loads_travel_plan_dict(m.group(1))
            if parsed is not None:
                return parsed

    parsed = _loads_travel_plan_dict(stripped)
    if parsed is not None:
        return parsed

    return _loads_embedded_json_object(stripped)


def merge_trip_inputs_from_history(
    base: dict[str, str],
    normalized_message_history: list[dict[str, str]],
) -> dict[str, str]:
    """
    On follow-up turns, form fields are often left blank. Pull destination / duration hints
    from the last assistant JSON so structured hints are not empty.
    """
    out = {k: (v or "").strip() for k, v in base.items()}
    need_dest = not out.get("destination")
    need_dates = not out.get("dates_or_duration")
    # Do not auto-fill budget from prior plan — follow-ups like "make it low-cost" must win.

    last_json: dict[str, Any] | None = None
    for m in reversed(normalized_message_history):
        if m.get("role") != "assistant":
            continue
        content = m.get("content") or ""
        last_json = _extract_json_block(content)
        if last_json:
            break

    if not last_json:
        return out

    dest = last_json.get("destination") or {}
    itin = last_json.get("itinerary") or {}

    if need_dest:
        summary = dest.get("destination_summary")
        if isinstance(summary, str) and summary.strip():
            out["destination"] = summary.strip()[:400]

    if need_dates:
        days = itin.get("days")
        if isinstance(days, list) and days:
            out["dates_or_duration"] = f"approximately {len(days)} days (from prior plan)"
        title = itin.get("trip_title")
        if isinstance(title, str) and title.strip() and not out.get("dates_or_duration"):
            out["dates_or_duration"] = title.strip()[:200]

    return out


def wrap_refinement_user_message(
    user_message: str,
    conversation_context: str,
) -> str:
    """Make follow-ups explicit for the LLM when session context exists."""
    msg = user_message.strip()
    if not msg:
        return msg
    if not conversation_context or conversation_context.strip() in ("(none)", ""):
        return msg
    return (
        "[Refinement / follow-up — keep the same destination and trip length from the prior plan "
        "unless the user explicitly changes them. Apply the new constraint below.]\n"
        f"{msg}"
    )
