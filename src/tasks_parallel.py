"""Single-task definitions for parallel orchestration (no Task.context chain)."""

from __future__ import annotations

from crewai import Task

from schemas import (
    BudgetPlanOutput,
    DestinationResearchOutput,
    HotelFinderOutput,
    ItineraryOutput,
)
from tasks import TRIP_CONTEXT

# Injected after phase 1 / 2 for prompt interpolation (defaults "(none)" in inputs).
INJECT = """
---
## Prior JSON from pipeline (parallel mode — do not invent; use when provided)
**Destination research JSON:**
{prior_destination_json}

**Budget plan JSON (may be unavailable during hotel-only phase):**
{prior_budget_json}

**Hotel recommendations JSON (may be unavailable until after hotel phase):**
{prior_hotels_json}
"""


def create_parallel_tasks(agents: dict[str, object]) -> dict[str, Task]:
    a = agents

    dest_task = Task(
        description=TRIP_CONTEXT
        + INJECT
        + """
## Your task (Destination Researcher)
Research and summarize the trip destination scope implied above. Cover climate/best season,
concrete highlights, and practical notes (transport, visas, safety, packing) relevant to this traveler.
If the user is refining a prior plan (see conversation), apply those constraints.
""",
        expected_output="A DestinationResearchOutput Pydantic object matching the schema exactly.",
        agent=a["destination_researcher"],
        output_pydantic=DestinationResearchOutput,
    )

    budget_task = Task(
        description=TRIP_CONTEXT
        + INJECT
        + """
## Your task (Budget Planner)
You have **destination research JSON** above (from the prior step). Use it as ground truth for location and trip style.
1) Read **budget_hint** and the user message as the authority on spend.
2) Set **stated_budget_interpretation** clearly.
3) Pick **overall_band** low/medium/high that matches that interpretation for this destination.
4) Give **daily_spend_estimate_notes** and **category_breakdown** accordingly.
""",
        expected_output="A BudgetPlanOutput Pydantic object matching the schema exactly.",
        agent=a["budget_planner"],
        output_pydantic=BudgetPlanOutput,
    )

    hotel_task = Task(
        description=TRIP_CONTEXT
        + INJECT
        + """
## Your task (Hotel Finder) — parallel mode note
**Budget Planner JSON may be empty here** because it runs in parallel with this task.
Align nightly rates strictly to the user's **budget_hint** and the **destination research JSON** above.
Recommend **at least two** named real properties (hotel/hostel/ryokan) with **approximate_nightly_price_hint**
matching that budget. Use **match_to_user_budget** to reference the user's stated budget (not a separate budget agent output).
If search tools exist, sanity-check property names for the destination.
""",
        expected_output="A HotelFinderOutput Pydantic object matching the schema exactly.",
        agent=a["hotel_finder"],
        output_pydantic=HotelFinderOutput,
    )

    itinerary_task = Task(
        description=TRIP_CONTEXT
        + INJECT
        + """
## Your task (Itinerary Generator)
You have **full JSON** for destination, budget, and hotels above. Build a coherent day-by-day plan that matches
**travel_style** and the **budget plan's overall_band**. Fill **budget_alignment_summary** first.
Respect **dates_or_duration** for the number of DayPlan entries. Reference hotel neighborhoods when routing days.
""",
        expected_output="An ItineraryOutput Pydantic object matching the schema exactly.",
        agent=a["itinerary_generator"],
        output_pydantic=ItineraryOutput,
    )

    return {
        "destination": dest_task,
        "budget": budget_task,
        "hotel": hotel_task,
        "itinerary": itinerary_task,
    }
