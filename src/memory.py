"""Session memory helpers: format Gradio (or CLI) history for task injection."""

from __future__ import annotations

# Remember last N user/assistant pairs to cap prompt size (full JSON excerpts).
MAX_ASSISTANT_EXCERPT_CHARS = 1500
MAX_TURNS_IN_CONTEXT = 8


def format_conversation_context(history: list) -> str:
    """
    Turn UI chat history into a single string for `{conversation_context}`.

    Supports Gradio 6+ **messages** format: `[{role, content}, ...]` (user/assistant
    alternating), and legacy **tuples** format: `[[user, bot], ...]`.
    """
    if not history:
        return "(none)"

    turns: list[str] = []

    # Messages format: each item is one role; last N turns ≈ 2 * MAX_TURNS_IN_CONTEXT msgs
    first = history[0]
    if isinstance(first, dict) and "role" in first and "content" in first:
        slice_hist = history[-(2 * MAX_TURNS_IN_CONTEXT) :]
        for m in slice_hist:
            if not isinstance(m, dict) or "role" not in m:
                continue
            role = m["role"]
            content = (m.get("content") or "").strip()
            if role == "assistant":
                excerpt = content[:MAX_ASSISTANT_EXCERPT_CHARS]
                if len(content) > MAX_ASSISTANT_EXCERPT_CHARS:
                    excerpt += "\n[…truncated…]"
                turns.append(f"ASSISTANT: {excerpt}")
            else:
                turns.append(f"{role.upper()}: {content}")
    else:
        slice_hist = history[-MAX_TURNS_IN_CONTEXT :]
        for turn in slice_hist:
            if isinstance(turn, dict) and "role" in turn:
                role = turn["role"]
                content = (turn.get("content") or "").strip()
                turns.append(f"{role.upper()}: {content}")
            elif isinstance(turn, (list, tuple)) and len(turn) >= 2:
                user_msg = (turn[0] or "").strip()
                bot_msg = (turn[1] or "").strip()
                excerpt = bot_msg[:MAX_ASSISTANT_EXCERPT_CHARS]
                if len(bot_msg) > MAX_ASSISTANT_EXCERPT_CHARS:
                    excerpt += "\n[…truncated…]"
                turns.append(f"User: {user_msg}\nAssistant (excerpt): {excerpt}")
            else:
                turns.append(str(turn))

    return "\n\n---\n\n".join(turns) if turns else "(none)"


def default_trip_inputs() -> dict[str, str]:
    """Empty structured slots; main/Gradio merge with user message."""
    return {
        "user_message": "",
        "conversation_context": "(none)",
        "destination": "",
        "dates_or_duration": "",
        "budget_hint": "",
        "interests": "",
        "travel_style": "",
        "prior_destination_json": "(none)",
        "prior_budget_json": "(none)",
        "prior_hotels_json": "(none)",
        # Full prior JSON report from UI for refine mode (skip re-running destination LLM).
        "stashed_prior_plan_json": "(none)",
    }
