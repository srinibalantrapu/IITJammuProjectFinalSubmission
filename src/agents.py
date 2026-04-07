"""Agent definitions: Destination Researcher, Budget Planner, Hotel Finder, Itinerary Generator."""

from __future__ import annotations

import functools
import os

from crewai import Agent


@functools.lru_cache(maxsize=1)
def _serper_tools_cached() -> tuple:
    """Build Serper tool once; returns tuple so it's hashable for the cache."""
    key = os.getenv("SERPER_API_KEY")
    if not key:
        return ()
    try:
        from crewai_tools import SerperDevTool
        return (SerperDevTool(),)
    except Exception:
        return ()


def _serper_tools_disabled() -> bool:
    return os.getenv("TRAVEL_PLANNER_DISABLE_SERPER", "").lower() in ("1", "true", "yes")


@functools.lru_cache(maxsize=1)
def _agent_config() -> dict:
    """Read env-based agent config once; returns model name and max_iter."""
    mdl = os.getenv("TRAVEL_PLANNER_MODEL") or os.getenv("OPENAI_MODEL") or None
    raw = os.getenv("TRAVEL_PLANNER_MAX_ITER", "10")
    try:
        mi: int | None = max(4, min(25, int(raw)))
    except ValueError:
        mi = None
    return {"llm": mdl, "max_iter": mi}


def create_agents(*, verbose: bool = False) -> dict[str, Agent]:
    serper = [] if _serper_tools_disabled() else list(_serper_tools_cached())
    dest_tools = serper
    hotel_tools = serper

    cfg = _agent_config()
    llm_kw: dict = {}
    if cfg["llm"]:
        llm_kw["llm"] = cfg["llm"]
    if cfg["max_iter"] is not None:
        llm_kw["max_iter"] = cfg["max_iter"]

    destination_researcher = Agent(
        role="Destination Researcher",
        goal=(
            "Summarize the destination: seasonality, highlights, and practical constraints "
            "aligned with the traveler's message, structured hints, and any prior chat."
        ),
        backstory=(
            "You are a meticulous travel researcher who blends geography, culture, and "
            "logistics. When search tools exist, you use them for timely facts; otherwise "
            "you label assumptions clearly."
        ),
        tools=dest_tools,
        allow_delegation=False,
        verbose=verbose,
        **llm_kw,
    )

    budget_planner = Agent(
        role="Budget Planner",
        goal=(
            "Translate the traveler's stated budget_hint and message into a clear low/medium/high band, "
            "numeric or daily/total interpretation in stated_budget_interpretation, and category breakdown."
        ),
        backstory=(
            "You always reconcile destination cost levels with the user's explicit budget words "
            "(e.g. USD per day, total trip cap, backpacker vs luxury). Downgrade activities before "
            "ignoring the budget hint."
        ),
        allow_delegation=False,
        verbose=verbose,
        **llm_kw,
    )

    hotel_finder = Agent(
        role="Hotel Finder",
        goal=(
            "Recommend **specific named hotels/hostels/ryokans** (real searchable property names) that fit "
            "the Budget Planner's band and destination; give nightly price hints aligned to that budget."
        ),
        backstory=(
            "You name actual properties travelers can type into Google Maps or booking sites -- never only "
            "generic categories. When web search tools exist, use them to cross-check names and neighborhoods; "
            "otherwise use well-known representative properties and still give concrete names. "
            "Disclaimer on rates remains mandatory."
        ),
        tools=hotel_tools,
        allow_delegation=False,
        verbose=verbose,
        **llm_kw,
    )

    itinerary_generator = Agent(
        role="Itinerary Generator",
        goal=(
            "Produce a day-by-day plan explicitly tuned to the budget band: choice of sights (free vs paid), "
            "meal tiers, and transport; reference hotel neighborhoods when it helps routing."
        ),
        backstory=(
            "You never write a champagne itinerary on a backpacker budget. You spell out trade-offs in "
            "budget_alignment_summary and keep daily blocks realistic for the stated spend level."
        ),
        allow_delegation=False,
        verbose=verbose,
        **llm_kw,
    )

    return {
        "destination_researcher": destination_researcher,
        "budget_planner": budget_planner,
        "hotel_finder": hotel_finder,
        "itinerary_generator": itinerary_generator,
    }
