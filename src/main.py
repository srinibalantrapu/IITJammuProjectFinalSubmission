"""CLI: run travel crew and print structured JSON report."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from dotenv import load_dotenv

# Allow `python src/main.py` from repo root without PYTHONPATH
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv(os.path.join(_ROOT, ".env"))

from crewai.crews.crew_output import CrewOutput

from crew import build_crew
from memory import default_trip_inputs
from parallel_pipeline import PipelineError, run_parallel_pipeline
from utils import dump_model


def aggregate_report(result: CrewOutput) -> dict[str, Any]:
    stage_keys = ["destination", "budget", "hotels", "itinerary"]
    out: dict[str, Any] = {}
    for i, key in enumerate(stage_keys):
        if i < len(result.tasks_output):
            to = result.tasks_output[i]
            out[key] = dump_model(getattr(to, "pydantic", None))
        else:
            out[key] = {}
    out["raw_final"] = result.raw
    return out


def _parallel_default() -> bool:
    return os.getenv("TRAVEL_PLANNER_PARALLEL", "1").lower() not in ("0", "false", "no")


def run_pipeline(
    inputs: dict[str, str],
    *,
    verbose: bool = False,
    unified_memory: bool | None = None,
    parallel: bool | None = None,
) -> tuple[dict[str, Any], CrewOutput | None]:
    merged = default_trip_inputs()
    merged.update({k: (v or "").strip() for k, v in inputs.items()})
    if not merged.get("conversation_context"):
        merged["conversation_context"] = "(none)"

    use_parallel = _parallel_default() if parallel is None else parallel
    if use_parallel:
        report = run_parallel_pipeline(
            merged,
            verbose=verbose,
            unified_memory=unified_memory,
        )
        return report, None

    crew = build_crew(verbose=verbose, unified_memory=unified_memory)
    result = crew.kickoff(inputs=merged)
    if not isinstance(result, CrewOutput):
        raise TypeError("Expected CrewOutput from kickoff")
    report = aggregate_report(result)
    report["pipeline_mode"] = "sequential"
    return report, result


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in .env or the environment.")

    parser = argparse.ArgumentParser(description="CrewAI AI Travel Planner (CLI)")
    parser.add_argument(
        "message",
        nargs="?",
        default="",
        help="Free-form trip request",
    )
    parser.add_argument("--destination", default="", help="Destination / region")
    parser.add_argument("--dates", default="", help="Dates or duration, e.g. 5 days in March")
    parser.add_argument("--budget", default="", help="Budget hint, e.g. low-cost, mid-range")
    parser.add_argument("--interests", default="", help="Interests, comma-separated")
    parser.add_argument("--style", default="", help="Travel style, e.g. relaxed, packed")
    parser.add_argument(
        "--context",
        default="",
        help="Simulated prior conversation (for testing memory injection)",
    )
    parser.add_argument(
        "--unified-memory",
        action="store_true",
        help="Enable CrewAI unified memory (slower; uses embeddings). Session context always works without this.",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Use one Crew with four sequential tasks (slower). Default: parallel pipeline (destination, then budget∥hotel, then itinerary).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    user_message = args.message.strip()
    if not user_message:
        user_message = (
            f"Plan a trip to {args.destination or 'a destination of your choice'} "
            f"for {args.dates or 'a few days'}."
        ).strip()

    inputs = {
        "user_message": user_message,
        "conversation_context": args.context.strip() or "(none)",
        "destination": args.destination,
        "dates_or_duration": args.dates,
        "budget_hint": args.budget,
        "interests": args.interests,
        "travel_style": args.style,
    }

    report, _ = run_pipeline(
        inputs,
        verbose=args.verbose,
        unified_memory=args.unified_memory,
        parallel=not args.sequential,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
