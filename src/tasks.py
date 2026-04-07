"""Tasks with structured Pydantic outputs and sequential context chain."""

from __future__ import annotations

from crewai import Task

from schemas import (
    BudgetPlanOutput,
    DestinationResearchOutput,
    HotelFinderOutput,
    ItineraryOutput,
)

TRIP_CONTEXT = """
## Current trip request
**User message:** {user_message}

**Structured hints** (use when provided; otherwise infer from the message and conversation):
- Destination / region: {destination}
- Dates or duration: {dates_or_duration}
- **Traveler budget (primary):** {budget_hint}  ← Always honor this when setting cost bands, hotels, and daily activities.
- Interests: {interests}
- Travel style: {travel_style}

**Follow-up rule:** If the section below contains a prior assistant plan, the user may be issuing a
**refinement only** (e.g. "Now make it low-cost"). In that case, **keep the same destination and trip length**
unless they explicitly change them. Apply the new constraint across budget, hotels, and itinerary.

## Conversation so far (session memory — includes prior plans and user follow-ups)
{conversation_context}
"""


def create_tasks(agents: dict[str, object]) -> list[Task]:
    a = agents

    dest_task = Task(
        description=TRIP_CONTEXT
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
        + """
## Your task (Budget Planner)
Using the destination research output as ground truth for location and trip style, build a budget plan:
1) Read **budget_hint** and the user message as the authority on spend (daily cap, total trip budget, or words like backpacker/mid/luxury).
2) Set **stated_budget_interpretation** to spell out that interpretation in plain language (include currency if inferable).
3) Pick **overall_band** low/medium/high that matches that interpretation for this destination.
4) Give **daily_spend_estimate_notes** and **category_breakdown** that stay inside that interpretation.
5) If the user asked to reduce cost in follow-up messages, shift the band and assumptions accordingly.
""",
        expected_output="A BudgetPlanOutput Pydantic object matching the schema exactly.",
        agent=a["budget_planner"],
        context=[dest_task],
        output_pydantic=BudgetPlanOutput,
    )

    hotel_task = Task(
        description=TRIP_CONTEXT
        + """
## Your task (Hotel Finder)
Using destination context and the **budget plan's band and stated_budget_interpretation**:
- Recommend **at least three** lodging options. Each must be a **specific named property** (real hotel, hostel, ryokan, or serviced apartment brand/building name travelers can search—not generic labels like "nice hotel").
- For each, fill **approximate_nightly_price_hint** so it fits the user's budget (same currency style as the budget step when possible).
- Use **match_to_user_budget** to tie each pick explicitly to the budget band.
- If search tools are available, use them to sanity-check names for the destination; otherwise use well-known representative properties.
- Keep **data_disclaimer** honest about verifying rates and availability before booking.
""",
        expected_output="A HotelFinderOutput Pydantic object matching the schema exactly.",
        agent=a["hotel_finder"],
        context=[dest_task, budget_task],
        output_pydantic=HotelFinderOutput,
    )

    itinerary_task = Task(
        description=TRIP_CONTEXT
        + """
## Your task (Itinerary Generator)
Build a day-by-day itinerary whose pacing matches **travel_style** and the **budget plan's overall_band**:
- Fill **budget_alignment_summary** first: explain how meals, transport (walk/transit vs taxi), and paid vs free sights match the user's budget.
- Each **DayPlan** should implicitly respect that band (e.g. low = more free walks, markets, public transit; high may include premium experiences if budget allows).
- The number of days must align with dates_or_duration when specified (e.g. "5 days" -> 5 DayPlan entries);
  if only vague duration is given, assume a sensible default and state it in pacing_notes.
- Reference highlights from destination research and **hotel neighborhoods** from the hotel step when routing days.
""",
        expected_output="An ItineraryOutput Pydantic object matching the schema exactly.",
        agent=a["itinerary_generator"],
        context=[dest_task, budget_task, hotel_task],
        output_pydantic=ItineraryOutput,
    )

    return [dest_task, budget_task, hotel_task, itinerary_task]
