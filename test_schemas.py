"""Validate that Pydantic output schemas instantiate and serialize correctly."""

from __future__ import annotations

import json

import pytest

from schemas import (
    BudgetLine,
    BudgetPlanOutput,
    DayPlan,
    DestinationResearchOutput,
    HotelFinderOutput,
    HotelRecommendation,
    ItineraryOutput,
)


def test_destination_research_output_roundtrip():
    obj = DestinationResearchOutput(
        destination_summary="5 days in Kyoto, Japan",
        climate_and_best_season="Cherry blossom season (late March - early April)",
        highlights=["Fushimi Inari", "Kinkaku-ji", "Arashiyama Bamboo"],
        practical_notes=["JR Pass recommended", "Cash still common"],
    )
    d = obj.model_dump()
    assert d["destination_summary"] == "5 days in Kyoto, Japan"
    assert len(d["highlights"]) == 3
    reloaded = DestinationResearchOutput.model_validate(json.loads(json.dumps(d)))
    assert reloaded == obj


def test_budget_plan_output_band_validation():
    obj = BudgetPlanOutput(
        currency_assumption="USD",
        overall_band="low",
        stated_budget_interpretation="$50/day backpacker budget",
        daily_spend_estimate_notes="Hostels, street food, walking",
        category_breakdown=[BudgetLine(category="Accommodation", notes_for_low_mid_high="$15-25/night dorm")],
        key_assumptions=["Dorm beds available", "Eating at konbini"],
    )
    assert obj.overall_band == "low"
    with pytest.raises(Exception):
        BudgetPlanOutput(
            currency_assumption="USD",
            overall_band="ultra-luxury",
            stated_budget_interpretation="x",
            daily_spend_estimate_notes="x",
            category_breakdown=[],
            key_assumptions=[],
        )


def test_hotel_finder_min_recommendations():
    rec = HotelRecommendation(
        property_name="Hotel Granvia Kyoto",
        property_type="hotel",
        neighborhood_or_area="Kyoto Station",
        approximate_nightly_price_hint="$120-150",
        vibe_and_amenities="Business hotel, free wifi, breakfast buffet",
        match_to_user_budget="Mid-range: fits $120/day budget",
    )
    obj = HotelFinderOutput(recommendations=[rec, rec])
    assert len(obj.recommendations) == 2

    with pytest.raises(Exception):
        HotelFinderOutput(recommendations=[rec] * 7)


def test_itinerary_day_number_ge1():
    with pytest.raises(Exception):
        DayPlan(day_number=0, theme="x", morning="x", afternoon="x", evening="x", local_tips="x")

    day = DayPlan(day_number=1, theme="Temples", morning="Fushimi Inari", afternoon="Kinkaku-ji", evening="Gion walk", local_tips="Wear comfy shoes")
    assert day.day_number == 1


def test_itinerary_output_roundtrip():
    day = DayPlan(day_number=1, theme="Culture", morning="m", afternoon="a", evening="e", local_tips="t")
    obj = ItineraryOutput(
        trip_title="5 Days in Kyoto",
        budget_alignment_summary="Low-cost: free temples, konbini meals",
        pacing_notes="Relaxed pace",
        days=[day],
    )
    d = json.loads(json.dumps(obj.model_dump()))
    assert d["days"][0]["theme"] == "Culture"
    reloaded = ItineraryOutput.model_validate(d)
    assert reloaded.trip_title == obj.trip_title
