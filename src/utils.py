"""Shared helpers used across pipeline, main, and Gradio."""

from __future__ import annotations

from typing import Any, NamedTuple, TypedDict


class TripInputs(TypedDict, total=False):
    """Structured trip input slots passed through the pipeline."""

    user_message: str
    conversation_context: str
    destination: str
    dates_or_duration: str
    budget_hint: str
    interests: str
    travel_style: str
    prior_destination_json: str
    prior_budget_json: str
    prior_hotels_json: str
    stashed_prior_plan_json: str


class TurnResult(NamedTuple):
    """Return type for _run_turn in the Gradio UI layer."""

    history: list[dict[str, str]]
    cleared_msg: str
    report: dict[str, Any]
    markdown: str
    new_state: dict[str, Any] | None
    status_text: str


def dump_model(obj: Any) -> dict[str, Any]:
    """Safely convert a Pydantic model (or None / raw object) to a plain dict."""
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return {"raw": str(obj)}


def content_to_str(content: object) -> str:
    """Flatten Gradio message content (str, list of parts, None) to a plain string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, dict) and p.get("text"):
                parts.append(str(p["text"]))
            elif isinstance(p, str):
                parts.append(p)
        return "\n".join(parts) if parts else str(content)
    return str(content)


def normalize_chat_history(history: list | None) -> list[dict[str, str]]:
    """
    Normalize to OpenAI-style message dicts (role + content).

    Handles Gradio 5+ messages format, ChatMessage objects, and legacy tuples.
    """
    history = history or []
    if not history:
        return []

    first = history[0]

    if isinstance(first, dict) and "role" in first and "content" in first:
        out: list[dict[str, str]] = []
        for m in history:
            if isinstance(m, dict) and "role" in m and "content" in m:
                out.append({"role": str(m["role"]), "content": content_to_str(m.get("content"))})
        return out

    if hasattr(first, "role") and hasattr(first, "content"):
        out = []
        for m in history:
            if hasattr(m, "role") and hasattr(m, "content"):
                out.append({"role": str(m.role), "content": content_to_str(m.content)})
        return out

    out = []
    for turn in history:
        if isinstance(turn, (list, tuple)) and len(turn) >= 2:
            out.append({"role": "user", "content": content_to_str(turn[0])})
            out.append({"role": "assistant", "content": content_to_str(turn[1])})
    return out
