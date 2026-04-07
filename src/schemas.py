"""Pydantic schemas for structured task outputs.

Each schema corresponds to one agent's output in the pipeline:

    DestinationResearchOutput  (Destination Researcher, phase 1)
        -> BudgetPlanOutput    (Budget Planner, phase 2)
        -> HotelFinderOutput   (Hotel Finder, phase 2 -- parallel with budget)
            -> ItineraryOutput (Itinerary Generator, phase 3 -- sees all prior outputs)
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DestinationResearchOutput(BaseModel):
    """Phase 1 output: destination overview, climate, highlights, and logistics."""

    destination_summary: str = Field(description="Overview of the place(s) and trip scope")
    climate_and_best_season: str
    highlights: list[str] = Field(description="Must-see or must-do items")
    practical_notes: list[str] = Field(
        description="Visa, transport, safety, or packing notes as applicable"
    )


class BudgetLine(BaseModel):
    """Single row in the budget category breakdown."""

    category: str
    notes_for_low_mid_high: str = Field(
        description="How costs might look across budget bands"
    )


class BudgetPlanOutput(BaseModel):
    """Phase 2 output: budget interpretation, band, and category breakdown."""

    currency_assumption: str
    overall_band: Literal["low", "medium", "high"]
    stated_budget_interpretation: str = Field(
        description="How the user's budget_hint and message were translated into this plan (e.g. daily cap, total trip ceiling)"
    )
    daily_spend_estimate_notes: str
    category_breakdown: list[BudgetLine]
    key_assumptions: list[str]


class HotelRecommendation(BaseModel):
    """A single named hotel/hostel/ryokan recommendation."""

    property_name: str = Field(
        description=(
            "A specific, searchable property name (real hotel/hostel/ryokan brand or building name "
            "typical for the destination). Do not answer with only a generic category like '3-star hotel'."
        )
    )
    property_type: str = Field(
        description="One of: hotel, hostel, guesthouse, ryokan, resort, serviced_apartment, boutique"
    )
    neighborhood_or_area: str
    approximate_nightly_price_hint: str = Field(
        description="Rough nightly rate range in local currency or USD/EUR that fits the user's budget band"
    )
    vibe_and_amenities: str = Field(description="Style and key amenities (wifi, breakfast, onsen, etc.)")
    match_to_user_budget: str = Field(
        description="One sentence linking this pick to the Budget Planner's band and the user's budget_hint"
    )


class HotelFinderOutput(BaseModel):
    """Phase 2 output: 2-6 named property recommendations with a data disclaimer."""

    data_disclaimer: str = Field(
        default="Verify names, prices, and availability before booking; rates are indicative.",
        description="Must state figures are indicative unless a live booking API was used",
    )
    recommendations: list[HotelRecommendation] = Field(
        min_length=2,
        max_length=6,
        description="At least two distinct named properties across suitable neighborhoods",
    )


class DayPlan(BaseModel):
    """Single day within the itinerary."""

    day_number: int = Field(ge=1)
    theme: str
    morning: str
    afternoon: str
    evening: str
    local_tips: str


class ItineraryOutput(BaseModel):
    """Phase 3 output: day-by-day plan aligned to budget, destination, and hotels."""

    trip_title: str
    budget_alignment_summary: str = Field(
        description=(
            "How this itinerary matches the budget band: paid vs free activities, meal level, "
            "transit vs taxis, and trade-offs made for the user's stated budget"
        )
    )
    pacing_notes: str
    days: list[DayPlan]
