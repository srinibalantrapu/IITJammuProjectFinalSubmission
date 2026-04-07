"""Gradio UI for the AI Travel Planner."""

from __future__ import annotations

import json
import logging
import os
import sys

import gradio as gr
from dotenv import load_dotenv

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from followup_context import merge_trip_inputs_from_history, wrap_refinement_user_message
from main import run_pipeline
from memory import format_conversation_context
from parallel_pipeline import PipelineError
from utils import TurnResult, normalize_chat_history

log = logging.getLogger(__name__)

MAX_USER_MESSAGE_CHARS = 4000

CSS = """
.header-banner {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 8px;
    color: white;
    text-align: center;
}
.header-banner h1 {
    margin: 0 0 6px 0;
    font-size: 2rem;
    font-weight: 700;
    color: white !important;
    letter-spacing: -0.5px;
}
.header-banner p {
    margin: 0;
    font-size: 1rem;
    opacity: 0.92;
    color: white !important;
}
.api-ok {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 10px;
    padding: 10px 16px;
    text-align: center;
    font-size: 0.9rem;
    color: #166534;
}
.api-missing {
    background: #fef2f2;
    border: 2px solid #fca5a5;
    border-radius: 12px;
    padding: 18px 20px;
    color: #991b1b;
}
.api-missing h3 { margin: 0 0 8px 0; color: #991b1b; }
.api-missing ol { margin: 4px 0 0 0; padding-left: 1.2rem; line-height: 1.7; color: #1f2937; }
.api-missing code { background: #fee2e2; padding: 1px 5px; border-radius: 4px; font-size: 0.88em; }
#pipeline-status {
    min-height: 28px;
    text-align: center;
    font-size: 0.95rem;
    padding: 4px 0;
}
#pipeline-status p { margin: 0; }
.hint-row .gr-textbox { min-height: 56px; }
footer { display: none !important; }
"""


def _openai_key_configured() -> bool:
    return bool((os.getenv("OPENAI_API_KEY") or "").strip())


def _api_status_html() -> str:
    if _openai_key_configured():
        return '<div class="api-ok">API key loaded &mdash; ready to plan trips.</div>'
    root = os.path.dirname(os.path.abspath(__file__))
    return f"""<div class="api-missing">
<h3>Setup required</h3>
<p><strong>OPENAI_API_KEY</strong> is not set. The agents cannot run until you add it.</p>
<ol>
<li>Copy <code>.env.example</code> to <code>.env</code> in <code>{root}</code></li>
<li>Add: <code>OPENAI_API_KEY=sk-...</code></li>
<li>Restart: <code>python app_gradio.py</code></li>
</ol>
</div>"""


def _on_app_load() -> None:
    if not _openai_key_configured():
        gr.Warning(
            "OPENAI_API_KEY is missing. Add it to .env and restart.",
            title="API key required",
            duration=0,
        )


def _append_messages(history: list | None, user_text: str, assistant_text: str) -> list[dict[str, str]]:
    out = normalize_chat_history(history)
    out.append({"role": "user", "content": user_text})
    out.append({"role": "assistant", "content": assistant_text})
    return out


def _report_to_markdown(report: dict) -> str:
    if report.get("error"):
        return f"**Error:** {report['error']}"

    lines: list[str] = []
    dest = report.get("destination") or {}
    if dest:
        lines.append(f"## {dest.get('destination_summary', 'Destination')}\n")
        lines.append(f"**Best season:** {dest.get('climate_and_best_season', 'N/A')}\n")
        if dest.get("highlights"):
            lines.append("**Highlights**")
            for h in dest["highlights"]:
                lines.append(f"- {h}")
            lines.append("")
        if dest.get("practical_notes"):
            lines.append("**Good to know**")
            for p in dest["practical_notes"]:
                lines.append(f"- {p}")
            lines.append("")

    bud = report.get("budget") or {}
    if bud:
        band = (bud.get("overall_band") or "").capitalize()
        curr = bud.get("currency_assumption", "")
        lines.append(f"## Budget: {band} ({curr})\n")
        if bud.get("stated_budget_interpretation"):
            lines.append(f"> {bud['stated_budget_interpretation']}\n")
        if bud.get("daily_spend_estimate_notes"):
            lines.append(f"{bud['daily_spend_estimate_notes']}\n")
        for row in bud.get("category_breakdown") or []:
            if isinstance(row, dict):
                lines.append(f"- **{row.get('category', '')}:** {row.get('notes_for_low_mid_high', '')}")
        lines.append("")

    hotels = report.get("hotels") or {}
    if hotels:
        lines.append("## Where to stay\n")
        disclaimer = hotels.get("data_disclaimer", "")
        if disclaimer:
            lines.append(f"*{disclaimer}*\n")
        for h in hotels.get("recommendations") or []:
            if not isinstance(h, dict):
                continue
            name = h.get("property_name", "Hotel")
            lines.append(f"### {name}")
            parts = []
            if h.get("property_type"):
                parts.append(h["property_type"].replace("_", " ").title())
            if h.get("neighborhood_or_area"):
                parts.append(h["neighborhood_or_area"])
            if parts:
                lines.append(f"_{' \u00b7 '.join(parts)}_\n")
            if h.get("approximate_nightly_price_hint"):
                lines.append(f"**Price:** {h['approximate_nightly_price_hint']}")
            if h.get("vibe_and_amenities"):
                lines.append(f"**Vibe:** {h['vibe_and_amenities']}")
            if h.get("match_to_user_budget"):
                lines.append(f"**Why this fits:** {h['match_to_user_budget']}")
            lines.append("")

    itin = report.get("itinerary") or {}
    if itin:
        title = itin.get("trip_title", "Your Itinerary")
        lines.append(f"## {title}\n")
        if itin.get("budget_alignment_summary"):
            lines.append(f"> **Budget fit:** {itin['budget_alignment_summary']}\n")
        if itin.get("pacing_notes"):
            lines.append(f"*{itin['pacing_notes']}*\n")
        for d in itin.get("days") or []:
            if not isinstance(d, dict):
                continue
            lines.append(f"### Day {d.get('day_number', '?')} \u2014 {d.get('theme', '')}")
            lines.append(f"**Morning:** {d.get('morning', '')}")
            lines.append(f"**Afternoon:** {d.get('afternoon', '')}")
            lines.append(f"**Evening:** {d.get('evening', '')}")
            if d.get("local_tips"):
                lines.append(f"\n*Tip: {d['local_tips']}*")
            lines.append("")

    return "\n".join(lines) if lines else "*No plan generated.*"


def _error_report(msg: str) -> dict:
    return {"error": msg, "destination": {}, "budget": {}, "hotels": {}, "itinerary": {}}


def _report_chat_summary(report: dict) -> str:
    if report.get("error"):
        return f"Something went wrong: {report['error']}"
    parts: list[str] = []
    dest = report.get("destination") or {}
    if dest.get("destination_summary"):
        parts.append(f"**Destination:** {dest['destination_summary'][:200]}")
    bud = report.get("budget") or {}
    if bud.get("overall_band"):
        parts.append(f"**Budget:** {bud['overall_band']} ({bud.get('currency_assumption', '')})")
    hotels = report.get("hotels") or {}
    names = [h.get("property_name", "") for h in (hotels.get("recommendations") or []) if isinstance(h, dict)]
    if names:
        parts.append(f"**Hotels:** {', '.join(names[:3])}")
    itin = report.get("itinerary") or {}
    if itin.get("trip_title"):
        n_days = len(itin.get("days") or [])
        parts.append(f"**Itinerary:** {itin['trip_title']} ({n_days} days)")
    parts.append("\n*Open the tabs below for the full plan.*")
    return "\n\n".join(parts)


def _run_turn(
    message: str,
    history: list,
    destination: str,
    dates: str,
    budget: str,
    interests: str,
    travel_style: str,
    prior_plan: dict | None,
) -> TurnResult:

    history = normalize_chat_history(history)
    conv = format_conversation_context(history)

    if not _openai_key_configured():
        err = _error_report("OPENAI_API_KEY is not set. Add it to .env and restart the app.")
        history = _append_messages(history, message or "(empty)", _report_chat_summary(err))
        return TurnResult(history, "", err, _report_to_markdown(err), prior_plan, "")

    message = (message or "").strip()[:MAX_USER_MESSAGE_CHARS]
    if not message:
        return TurnResult(history, "", {}, "", prior_plan, "")

    inputs: dict[str, str] = {
        "user_message": "",
        "conversation_context": conv,
        "destination": destination or "",
        "dates_or_duration": dates or "",
        "budget_hint": budget or "",
        "interests": interests or "",
        "travel_style": travel_style or "",
        "stashed_prior_plan_json": "(none)",
    }
    if prior_plan and isinstance(prior_plan, dict) and prior_plan.get("destination"):
        inputs["stashed_prior_plan_json"] = json.dumps(prior_plan, ensure_ascii=False)

    inputs = merge_trip_inputs_from_history(inputs, history)
    inputs["user_message"] = wrap_refinement_user_message(message, conv)

    low = ("low-cost", "low cost", "cheaper", "cheap", "budget", "econom", "affordable", "backpack")
    if not inputs.get("budget_hint") and any(k in message.lower() for k in low):
        inputs["budget_hint"] = message

    try:
        report, _ = run_pipeline(inputs, verbose=False, unified_memory=False)
    except (PipelineError, RuntimeError, TypeError) as e:
        log.exception("Pipeline failed")
        report = _error_report(str(e))
    except KeyboardInterrupt:
        raise
    except Exception as e:
        log.exception("Unexpected error in pipeline")
        report = _error_report(f"Unexpected error: {e}")

    md = _report_to_markdown(report)
    chat_summary = _report_chat_summary(report)
    history = _append_messages(history, message, chat_summary)
    if report.get("error"):
        return TurnResult(history, "", report, md, prior_plan, "")
    return TurnResult(history, "", report, md, report, "")


def launch() -> None:
    _theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.indigo,
        secondary_hue=gr.themes.colors.purple,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    )

    with gr.Blocks(title="AI Travel Planner") as demo:
        demo.load(_on_app_load, inputs=None, outputs=None)

        # ---------- header ----------
        gr.HTML(
            '<div class="header-banner">'
            "<h1>AI Travel Planner</h1>"
            "<p>Four AI agents collaborate to research your destination, plan your budget, "
            "find hotels, and build a day-by-day itinerary.</p>"
            "</div>"
        )
        gr.HTML(value=_api_status_html())

        # ---------- trip details ----------
        gr.Markdown("### Describe your trip")
        with gr.Row(equal_height=True):
            destination = gr.Textbox(
                label="Destination",
                placeholder="Kyoto, Japan",
                scale=2,
            )
            dates = gr.Textbox(
                label="Dates / duration",
                placeholder="5 days in late March",
                scale=2,
            )
            budget = gr.Textbox(
                label="Budget",
                placeholder="$120/day, mid-range, backpacker...",
                scale=2,
            )
        with gr.Row(equal_height=True):
            interests = gr.Textbox(
                label="Interests",
                placeholder="temples, street food, hiking...",
                scale=3,
            )
            travel_style = gr.Textbox(
                label="Style",
                placeholder="relaxed, packed, family-friendly",
                scale=2,
            )

        # ---------- chat + controls ----------
        status_label = gr.Markdown(value="", elem_id="pipeline-status")
        chat = gr.Chatbot(
            label="Conversation",
            height=400,
            placeholder="Your trip plan will appear here. Ask a follow-up to refine it.",
        )
        with gr.Row():
            msg = gr.Textbox(
                label="Message",
                placeholder="Plan my trip! (or send a follow-up like 'make it cheaper')",
                scale=5,
                show_label=False,
                container=False,
            )
            send_btn = gr.Button("Plan my trip", variant="primary", scale=1, min_width=140)

        with gr.Row():
            clear_btn = gr.Button("Start over", variant="secondary", size="sm")

        # ---------- results tabs ----------
        with gr.Tabs():
            with gr.Tab("Full plan"):
                md_view = gr.Markdown(value="*Submit a message above to generate your travel plan.*")
            with gr.Tab("Raw JSON"):
                json_view = gr.JSON(label="Structured report")
            with gr.Tab("Debug"):
                ctx_preview = gr.Markdown(value="*Session context will appear here after the first reply.*")

        # ---------- state ----------
        prior_plan = gr.State(None)
        inputs_list = [msg, chat, destination, dates, budget, interests, travel_style, prior_plan]

        def _show_running(m: str, h: list, prev: dict | None) -> tuple:
            h_norm = normalize_chat_history(h)
            conv = format_conversation_context(h_norm)
            is_refine = conv not in ("(none)", "") and prev
            if is_refine:
                label = "Refining your plan \u2014 reusing destination, updating budget, hotels & itinerary..."
            else:
                label = "Building your plan \u2014 researching destination, planning budget, finding hotels & creating itinerary..."
            return (
                f"<p style='text-align:center;color:#6366f1;font-weight:600'>{label}</p>",
                gr.update(interactive=False),
            )

        def submit(m: str, h: list, dest: str, dts: str, bud: str, intr: str, sty: str, prev: dict | None) -> tuple:
            result = _run_turn(m, h, dest, dts, bud, intr, sty, prev)
            preview = ""
            if h:
                nh = normalize_chat_history(h)
                pv = format_conversation_context(nh)
                preview = f"**Conversation context** (truncated):\n```\n{pv[:2500]}{'...' if len(pv) > 2500 else ''}\n```"
            return (
                result.history,
                "",
                result.report,
                result.markdown,
                result.new_state,
                preview,
                "",
                gr.update(interactive=True),
            )

        def clear() -> tuple:
            return (
                [],
                "",
                {},
                "*Submit a message above to generate your travel plan.*",
                None,
                "*Session context will appear here after the first reply.*",
                "",
                gr.update(interactive=True),
            )

        all_outputs = [chat, msg, json_view, md_view, prior_plan, ctx_preview, status_label, send_btn]

        msg.submit(
            _show_running,
            inputs=[msg, chat, prior_plan],
            outputs=[status_label, send_btn],
            show_progress="hidden",
        ).then(
            submit,
            inputs=inputs_list,
            outputs=all_outputs,
            show_progress="hidden",
        )

        send_btn.click(
            _show_running,
            inputs=[msg, chat, prior_plan],
            outputs=[status_label, send_btn],
            show_progress="hidden",
        ).then(
            submit,
            inputs=inputs_list,
            outputs=all_outputs,
            show_progress="hidden",
        )

        clear_btn.click(clear, outputs=all_outputs)

    demo.queue(default_concurrency_limit=1)
    demo.launch(theme=_theme, css=CSS)


if __name__ == "__main__":
    launch()
